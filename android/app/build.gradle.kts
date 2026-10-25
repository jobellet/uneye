// App module: the Kotlin UI + the OfflineSession JNI library.
//
// The native side is built by ../CMakeLists.txt (the frozen cpp/ tree plus
// android/offline_session + android/bench). The JNI bridge library is the
// `offline_session_jni` target, renamed to liboffline_session.so.
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.uneye.offline"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.uneye.offline"
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    externalNativeBuild {
        cmake {
            path = file("../CMakeLists.txt")
            version = "3.22.1"
        }
    }

    packaging {
        jniLibs { useLegacyPackaging = false }
    }
}

val syncWebAssets by tasks.registering(Copy::class) {
    description = "Synchronizes web app assets into Android assets for unified UI"
    from(rootProject.file("..")) {
        include("index.html")
        include("style.css")
        include("app.js")
        include("formats.js")
        include("import.js")
        include("import_ui.js")
        include("hmm_prelabel.js")
        include("train_worker.js")
        include("sw.js")
        include("logo.jpeg")
        include("logo_circle.png")
        include("web/**")
        include("models/**")
    }
    into(file("src/main/assets/web"))
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }.configureEach {
    dependsOn(syncWebAssets)
}
tasks.named("preBuild") {
    dependsOn(syncWebAssets)
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("androidx.activity:activity-ktx:1.9.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
    implementation("androidx.work:work-runtime-ktx:2.9.1")

    val roomVersion = "2.6.1"
    implementation("androidx.room:room-runtime:$roomVersion")
    implementation("androidx.room:room-ktx:$roomVersion")
    ksp("androidx.room:room-compiler:$roomVersion")

    implementation("androidx.webkit:webkit:1.12.1")

    testImplementation("junit:junit:4.13.2")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.work:work-testing:2.9.1")
}
