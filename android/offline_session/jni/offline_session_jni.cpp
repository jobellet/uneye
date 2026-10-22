// JNI bridge for the offline-labeling app. Exposes ONLY the OfflineSession
// facade (android/offline_session/include/offline_session/offline_session.hpp);
// no frozen contract is reached from Kotlin. The Kotlin side mirrors this
// surface in `com.uneye.offline.Native`.
//
// Threading: every entry point locks the facade internally. The training
// progress callback is invoked from the trainer's worker thread; it posts to
// the JVM via a cached method id and a global ref on the Kotlin listener,
// so Kotlin always receives progress on the thread it registered from.
#include <jni.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "offline_session/offline_session.hpp"

namespace {

constexpr const char* kClass = "com/uneye/offline/Native";

offline::OfflineSession* sess(jlong handle) {
    return reinterpret_cast<offline::OfflineSession*>(static_cast<uintptr_t>(handle));
}

std::string jstr(JNIEnv* env, jstring s) {
    if (!s) return {};
    const char* c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    if (c) env->ReleaseStringUTFChars(s, c);
    return out;
}

jstring to_jstr(JNIEnv* env, const std::string& s) {
    return env->NewStringUTF(s.c_str());
}

// --- training progress: posted to the JVM from the trainer worker thread ---

struct ProgressSink {
    JavaVM* vm = nullptr;
    jobject listener = nullptr;          // global ref on the Kotlin listener
    jmethodID on_progress = nullptr;     // void onProgress(int epoch, double trainLoss,
                                         //   double valLoss, double valF1, double kappa,
                                         //   double elapsedS, boolean finished,
                                         //   boolean cancelled, boolean failed, String message)
    std::atomic<bool> attached{false};
    std::mutex mu;
};

ProgressSink g_progress;

void post_progress(const offline::TrainEvent& e) {
    JavaVM* vm = g_progress.vm;
    if (!vm || !g_progress.listener || !g_progress.on_progress) return;
    JNIEnv* env = nullptr;
    const bool need_detach = g_progress.attached.exchange(true) == false;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        if (vm->AttachCurrentThread(&env, nullptr) != JNI_OK) { g_progress.attached = false; return; }
    }
    jstring msg = to_jstr(env, e.message);
    env->CallVoidMethod(g_progress.listener, g_progress.on_progress,
                        e.epoch, e.train_loss, e.val_loss, e.val_event_f1, e.val_kappa,
                        e.elapsed_s, e.finished ? JNI_TRUE : JNI_FALSE,
                        e.cancelled ? JNI_TRUE : JNI_FALSE, e.failed ? JNI_TRUE : JNI_FALSE, msg);
    if (env->ExceptionCheck()) env->ExceptionClear();   // never kill the trainer thread
    env->DeleteLocalRef(msg);
    if (need_detach) vm->DetachCurrentThread();
    g_progress.attached = false;
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_uneye_offline_Native_nativeCreate(JNIEnv*, jclass) {
    return static_cast<jlong>(reinterpret_cast<uintptr_t>(new offline::OfflineSession()));
}

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeDestroy(JNIEnv* env, jclass, jlong handle) {
    auto* s = sess(handle);
    if (!s) return;
    s->cancel_training();
    s->wait_training_done();
    delete s;
    if (g_progress.listener) { env->DeleteGlobalRef(g_progress.listener); g_progress.listener = nullptr; }
}

// load(xRows, yRows, rateHz, degPerUnit, estimateScale, prelabeler, modelBin) -> LoadResult as Object[]
// Object[] = { Boolean ok, String error, long nSamples, double rateHz, double degPerUnit,
//              Boolean scaleEstimated, Boolean hmmRateWarning, String prelabelerError,
//              int[] onset, int[] offset, int[] cls, int[] fromPrelabel }
JNIEXPORT jobjectArray JNICALL
Java_com_uneye_offline_Native_nativeLoad(JNIEnv* env, jclass, jlong handle,
                                         jobjectArray x_rows, jobjectArray y_rows,
                                         jdouble rate_hz, jdouble deg_per_unit,
                                         jboolean estimate_scale, jint prelabeler,
                                         jstring model_bin) {
    offline::OfflineSession* s = sess(handle);
    std::vector<std::vector<double>> X, Y;
    const jsize nx = env->GetArrayLength(x_rows);
    for (jsize r = 0; r < nx; ++r) {
        auto* xr = static_cast<jdoubleArray>(env->GetObjectArrayElement(x_rows, r));
        auto* yr = static_cast<jdoubleArray>(env->GetObjectArrayElement(y_rows, r));
        if (!xr || !yr) continue;
        const jsize n = env->GetArrayLength(xr);
        std::vector<double> x(n), y(env->GetArrayLength(yr));
        env->GetDoubleArrayRegion(xr, 0, n, x.data());
        env->GetDoubleArrayRegion(yr, 0, jsize(y.size()), y.data());
        env->DeleteLocalRef(xr); env->DeleteLocalRef(yr);
        X.push_back(std::move(x)); Y.push_back(std::move(y));
    }
    offline::LoadOptions opt;
    opt.rate_hz = rate_hz;
    opt.deg_per_unit = deg_per_unit;
    opt.estimate_scale = estimate_scale == JNI_TRUE;
    opt.prelabeler = static_cast<offline::Prelabeler>(prelabeler);
    opt.model_bin = jstr(env, model_bin);
    const offline::LoadResult res = s->load(X, Y, opt);

    const jsize nm = jsize(res.prelabels.size());
    auto* onset = env->NewIntArray(nm);
    auto* offset = env->NewIntArray(nm);
    auto* cls = env->NewIntArray(nm);
    auto* fromp = env->NewIntArray(nm);
    std::vector<jint> o(nm), f(nm), c(nm), p(nm);
    for (jsize i = 0; i < nm; ++i) {
        o[i] = jint(res.prelabels[size_t(i)].onset);
        f[i] = jint(res.prelabels[size_t(i)].offset);
        c[i] = res.prelabels[size_t(i)].cls;
        p[i] = res.prelabels[size_t(i)].from_prelabel ? 1 : 0;
    }
    env->SetIntArrayRegion(onset, 0, nm, o.data());
    env->SetIntArrayRegion(offset, 0, nm, f.data());
    env->SetIntArrayRegion(cls, 0, nm, c.data());
    env->SetIntArrayRegion(fromp, 0, nm, p.data());

    auto* out = env->NewObjectArray(13, env->FindClass("java/lang/Object"), nullptr);
    auto set = [&](int i, jobject v) { env->SetObjectArrayElement(out, i, v); if (v) env->DeleteLocalRef(v); };
    jclass boolc = env->FindClass("java/lang/Boolean");
    jclass longc = env->FindClass("java/lang/Long");
    jclass dblc = env->FindClass("java/lang/Double");
    jmethodID boolv = env->GetStaticMethodID(boolc, "valueOf", "(Z)Ljava/lang/Boolean;");
    jmethodID longv = env->GetStaticMethodID(longc, "valueOf", "(J)Ljava/lang/Long;");
    jmethodID dblv = env->GetStaticMethodID(dblc, "valueOf", "(D)Ljava/lang/Double;");
    set(0, env->CallStaticObjectMethod(boolc, boolv, res.ok ? JNI_TRUE : JNI_FALSE));
    set(1, to_jstr(env, res.error));
    set(2, env->CallStaticObjectMethod(longc, longv, jlong(res.n_samples)));
    set(3, env->CallStaticObjectMethod(dblc, dblv, res.rate_hz));
    set(4, env->CallStaticObjectMethod(dblc, dblv, res.deg_per_unit));
    set(5, env->CallStaticObjectMethod(boolc, boolv, res.scale_estimated ? JNI_TRUE : JNI_FALSE));
    set(6, env->CallStaticObjectMethod(boolc, boolv, res.hmm_rate_warning ? JNI_TRUE : JNI_FALSE));
    set(7, to_jstr(env, res.prelabeler_error));
    set(8, onset); set(9, offset); set(10, cls); set(11, fromp);
    set(12, to_jstr(env, ""));   // reserved
    env->DeleteLocalRef(boolc); env->DeleteLocalRef(longc); env->DeleteLocalRef(dblc);
    return out;
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeIsLoaded(JNIEnv*, jclass, jlong handle) {
    return sess(handle)->is_loaded() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL
Java_com_uneye_offline_Native_nativeNSamples(JNIEnv*, jclass, jlong handle) {
    return jlong(sess(handle)->n_samples());
}

JNIEXPORT jlong JNICALL
Java_com_uneye_offline_Native_nativeNTrials(JNIEnv*, jclass, jlong handle) {
    return jlong(sess(handle)->n_trials());
}

JNIEXPORT jdouble JNICALL
Java_com_uneye_offline_Native_nativeDegPerUnit(JNIEnv*, jclass, jlong handle) {
    return sess(handle)->deg_per_unit();
}

// trace(trial, from, to) -> float[2][len] : { x, y }
JNIEXPORT jobjectArray JNICALL
Java_com_uneye_offline_Native_nativeTrace(JNIEnv* env, jclass, jlong handle,
                                          jlong trial, jlong from, jlong to) {
    std::vector<float> x, y;
    sess(handle)->trace(trial, from, to, &x, &y);
    const jsize n = jsize(x.size());
    auto* xa = env->NewFloatArray(n); env->SetFloatArrayRegion(xa, 0, n, x.data());
    auto* ya = env->NewFloatArray(n); env->SetFloatArrayRegion(ya, 0, n, y.data());
    auto* out = env->NewObjectArray(2, env->FindClass("[F"), nullptr);
    env->SetObjectArrayElement(out, 0, xa);
    env->SetObjectArrayElement(out, 1, ya);
    env->DeleteLocalRef(xa); env->DeleteLocalRef(ya);
    return out;
}

// labels(trial, from, to, onlyReviewed) -> byte[len]
JNIEXPORT jbyteArray JNICALL
Java_com_uneye_offline_Native_nativeLabels(JNIEnv* env, jclass, jlong handle,
                                           jlong trial, jlong from, jlong to,
                                           jboolean only_reviewed) {
    std::vector<uint8_t> lab;
    sess(handle)->labels(trial, from, to, only_reviewed == JNI_TRUE, &lab);
    const jsize n = jsize(lab.size());
    auto* out = env->NewByteArray(n);
    env->SetByteArrayRegion(out, 0, n, reinterpret_cast<const jbyte*>(lab.data()));
    return out;
}

// --- pass 1: point marks -----------------------------------------------------

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeAddPointMark(JNIEnv*, jclass, jlong handle,
                                                 jlong trial, jlong at, jint cls) {
    sess(handle)->add_point_mark(trial, at, uint8_t(cls));
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeRemovePointMark(JNIEnv*, jclass, jlong handle,
                                                    jlong trial, jlong at) {
    return sess(handle)->remove_point_mark(trial, at) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeClearPointMarks(JNIEnv*, jclass, jlong handle, jlong trial) {
    sess(handle)->clear_point_marks(trial);
}

// pointMarks(trial) -> int[3][n] : { at, cls }
JNIEXPORT jobjectArray JNICALL
Java_com_uneye_offline_Native_nativePointMarks(JNIEnv* env, jclass, jlong handle, jlong trial) {
    const auto pm = sess(handle)->point_marks(trial);
    const jsize n = jsize(pm.size());
    auto* at = env->NewIntArray(n);
    auto* cl = env->NewIntArray(n);
    std::vector<jint> a(n), c(n);
    for (jsize i = 0; i < n; ++i) { a[i] = jint(pm[size_t(i)].at); c[i] = pm[size_t(i)].cls; }
    env->SetIntArrayRegion(at, 0, n, a.data());
    env->SetIntArrayRegion(cl, 0, n, c.data());
    auto* out = env->NewObjectArray(2, env->FindClass("[I"), nullptr);
    env->SetObjectArrayElement(out, 0, at);
    env->SetObjectArrayElement(out, 1, cl);
    env->DeleteLocalRef(at); env->DeleteLocalRef(cl);
    return out;
}

JNIEXPORT jlong JNICALL
Java_com_uneye_offline_Native_nativeSnapToPrelabel(JNIEnv*, jclass, jlong handle,
                                                   jlong trial, jlong at, jlong tolerance) {
    return jlong(sess(handle)->snap_to_prelabel(trial, at, tolerance));
}

// --- pass 2: refinement -------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeRefineMark(JNIEnv*, jclass, jlong handle,
                                               jlong trial, jlong at, jlong onset, jlong offset, jint cls) {
    return sess(handle)->refine_mark(trial, at, onset, offset, uint8_t(cls)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeDeleteMark(JNIEnv*, jclass, jlong handle, jlong trial, jlong at) {
    return sess(handle)->delete_mark(trial, at) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeMarkEpochReviewed(JNIEnv*, jclass, jlong handle,
                                                      jlong trial, jlong from, jlong to) {
    sess(handle)->mark_epoch_reviewed(trial, from, to);
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeUndo(JNIEnv*, jclass, jlong handle, jlong trial) {
    return sess(handle)->undo(trial) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeRedo(JNIEnv*, jclass, jlong handle, jlong trial) {
    return sess(handle)->redo(trial) ? JNI_TRUE : JNI_FALSE;
}

// events(trial) -> int[4][n] : { onset, offset, cls, fromPrelabel }
JNIEXPORT jobjectArray JNICALL
Java_com_uneye_offline_Native_nativeEvents(JNIEnv* env, jclass, jlong handle, jlong trial) {
    const auto ev = sess(handle)->events(trial);
    const jsize n = jsize(ev.size());
    std::vector<jint> o(n), f(n), c(n), p(n);
    for (jsize i = 0; i < n; ++i) {
        o[i] = jint(ev[size_t(i)].onset); f[i] = jint(ev[size_t(i)].offset);
        c[i] = ev[size_t(i)].cls; p[i] = ev[size_t(i)].from_prelabel ? 1 : 0;
    }
    auto mk = [&](const std::vector<jint>& v) {
        auto* a = env->NewIntArray(n); env->SetIntArrayRegion(a, 0, n, v.data()); return a;
    };
    auto* onset = mk(o); auto* offset = mk(f); auto* cls = mk(c); auto* fromp = mk(p);
    auto* out = env->NewObjectArray(4, env->FindClass("[I"), nullptr);
    env->SetObjectArrayElement(out, 0, onset);
    env->SetObjectArrayElement(out, 1, offset);
    env->SetObjectArrayElement(out, 2, cls);
    env->SetObjectArrayElement(out, 3, fromp);
    env->DeleteLocalRef(onset); env->DeleteLocalRef(offset);
    env->DeleteLocalRef(cls); env->DeleteLocalRef(fromp);
    return out;
}

// prelabels(trial) -> int[4][n] : { onset, offset, cls, fromPrelabel }
JNIEXPORT jobjectArray JNICALL
Java_com_uneye_offline_Native_nativePrelabels(JNIEnv* env, jclass, jlong handle, jlong trial) {
    const auto ev = sess(handle)->prelabels(trial);
    const jsize n = jsize(ev.size());
    std::vector<jint> o(n), f(n), c(n), p(n);
    for (jsize i = 0; i < n; ++i) {
        o[i] = jint(ev[size_t(i)].onset); f[i] = jint(ev[size_t(i)].offset);
        c[i] = ev[size_t(i)].cls; p[i] = ev[size_t(i)].from_prelabel ? 1 : 0;
    }
    auto mk = [&](const std::vector<jint>& v) {
        auto* a = env->NewIntArray(n); env->SetIntArrayRegion(a, 0, n, v.data()); return a;
    };
    auto* onset = mk(o); auto* offset = mk(f); auto* cls = mk(c); auto* fromp = mk(p);
    auto* out = env->NewObjectArray(4, env->FindClass("[I"), nullptr);
    env->SetObjectArrayElement(out, 0, onset);
    env->SetObjectArrayElement(out, 1, offset);
    env->SetObjectArrayElement(out, 2, cls);
    env->SetObjectArrayElement(out, 3, fromp);
    env->DeleteLocalRef(onset); env->DeleteLocalRef(offset);
    env->DeleteLocalRef(cls); env->DeleteLocalRef(fromp);
    return out;
}

JNIEXPORT jdouble JNICALL
Java_com_uneye_offline_Native_nativeReviewedSeconds(JNIEnv*, jclass, jlong handle) {
    return sess(handle)->reviewed_seconds();
}

// --- persistence --------------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeSaveLabels(JNIEnv* env, jclass, jlong handle, jstring path) {
    std::string err;
    const bool ok = sess(handle)->save_labels(jstr(env, path), &err);
    if (!ok && env) env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeLoadLabels(JNIEnv* env, jclass, jlong handle, jstring path) {
    std::string err;
    const bool ok = sess(handle)->load_labels(jstr(env, path), &err);
    if (!ok && env) env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

// --- training (phase 3) ---------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeStartTraining(JNIEnv* env, jclass, jlong handle,
                                                  jdouble max_seconds, jint n_threads,
                                                  jint max_epochs, jint patience,
                                                  jboolean rotation_aug,
                                                  jstring init_model,
                                                  jboolean use_last_model_as_init,
                                                  jobject listener) {
    if (g_progress.listener) { env->DeleteGlobalRef(g_progress.listener); g_progress.listener = nullptr; }
    if (listener) {
        g_progress.listener = env->NewGlobalRef(listener);
        jclass lc = env->GetObjectClass(listener);
        g_progress.on_progress = env->GetMethodID(lc, "onProgress",
            "(IDDDDDZZZLjava/lang/String;)V");
        env->DeleteLocalRef(lc);
        if (!g_progress.on_progress) { env->DeleteGlobalRef(g_progress.listener); g_progress.listener = nullptr; }
    }
    env->GetJavaVM(&g_progress.vm);
    offline::TrainRequest req;
    req.max_seconds = max_seconds;
    req.n_threads = n_threads;
    req.max_epochs = max_epochs;
    req.patience = patience;
    req.rotation_aug = rotation_aug == JNI_TRUE;
    req.init_model = jstr(env, init_model);
    req.use_last_model_as_init = use_last_model_as_init == JNI_TRUE;
    std::string err;
    const bool ok = sess(handle)->start_training(req, [](const offline::TrainEvent& e) { post_progress(e); }, &err);
    if (!ok) env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeCancelTraining(JNIEnv*, jclass, jlong handle) {
    sess(handle)->cancel_training();
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeTrainingRunning(JNIEnv*, jclass, jlong handle) {
    return sess(handle)->training_running() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeWaitTrainingDone(JNIEnv*, jclass, jlong handle) {
    sess(handle)->wait_training_done();
}

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativeSaveTrainedModel(JNIEnv* env, jclass, jlong handle, jstring path) {
    std::string err;
    const bool ok = sess(handle)->save_trained_model(jstr(env, path), &err);
    if (!ok) env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jstring JNICALL
Java_com_uneye_offline_Native_nativeLastTrainedModel(JNIEnv* env, jclass, jlong handle) {
    return to_jstr(env, sess(handle)->last_trained_model());
}

// --- convergence test set (phase 4) ----------------------------------------------

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeChooseTestSet(JNIEnv*, jclass, jlong handle,
                                                  jlong every_nth, jlong min_trials) {
    sess(handle)->choose_test_set(every_nth, min_trials);
}

// testTrials() -> long[]
JNIEXPORT jlongArray JNICALL
Java_com_uneye_offline_Native_nativeTestTrials(JNIEnv* env, jclass, jlong handle) {
    const auto t = sess(handle)->test_trials();
    const jsize n = jsize(t.size());
    auto* out = env->NewLongArray(n);
    std::vector<jlong> v(n);
    for (jsize i = 0; i < n; ++i) v[i] = jlong(t[size_t(i)]);
    env->SetLongArrayRegion(out, 0, n, v.data());
    return out;
}

// scoreTestSet(modelBin) -> double[11] : { ok(0/1), nSamples, f1[0..4], kappa[0..4] }
// (event counts per class are reported separately below)
JNIEXPORT jdoubleArray JNICALL
Java_com_uneye_offline_Native_nativeScoreTestSet(JNIEnv* env, jclass, jlong handle, jstring model_bin) {
    std::string err;
    const auto sc = sess(handle)->score_test_set(jstr(env, model_bin), &err);
    if (!sc.ok) env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err.c_str());
    auto* out = env->NewDoubleArray(11);
    std::vector<jdouble> v(11);
    v[0] = sc.ok ? 1.0 : 0.0;
    v[1] = double(sc.n_samples);
    for (int i = 0; i < 5; ++i) { v[2 + i] = sc.event_f1[i]; v[7 + i] = sc.kappa[i]; }
    env->SetDoubleArrayRegion(out, 0, 11, v.data());
    return out;
}

// scoreTestSetEventCounts(modelBin) -> int[5] : expert event counts per class on the test set
JNIEXPORT jintArray JNICALL
Java_com_uneye_offline_Native_nativeScoreTestSetEventCounts(JNIEnv* env, jclass, jlong handle, jstring model_bin) {
    std::string err;
    const auto sc = sess(handle)->score_test_set(jstr(env, model_bin), &err);
    auto* out = env->NewIntArray(5);
    jint v[5] = {0, 0, 0, 0, 0};
    if (sc.ok) for (int i = 0; i < 5; ++i) v[i] = sc.n_events[i];
    env->SetIntArrayRegion(out, 0, 5, v);
    return out;
}

// --- pre-labeling -----------------------------------------------------------------

JNIEXPORT jboolean JNICALL
Java_com_uneye_offline_Native_nativePrelabelTrial(JNIEnv* env, jclass, jlong handle, jlong trial) {
    std::string err;
    const bool ok = sess(handle)->prelabel_trial(trial, &err);
    if (!ok) env->ThrowNew(env->FindClass("java/lang/RuntimeException"), err.c_str());
    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_uneye_offline_Native_nativeSetPrelabeler(JNIEnv* env, jclass, jlong handle,
                                                  jint prelabeler, jstring model_bin) {
    sess(handle)->set_prelabeler(static_cast<offline::Prelabeler>(prelabeler), jstr(env, model_bin));
}

JNIEXPORT jint JNICALL
Java_com_uneye_offline_Native_nativePrelabeler(JNIEnv*, jclass, jlong handle) {
    return jint(sess(handle)->prelabeler());
}

}  // extern "C"
