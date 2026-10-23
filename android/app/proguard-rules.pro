# The JNI bridge is called through JNI name mangling: keep the Native object
# and its ProgressListener interface.
-keep class com.uneye.offline.Native { *; }
-keep interface com.uneye.offline.Native$ProgressListener { *; }
-keepclasseswithmembernames class com.uneye.offline.Native$ProgressListener { *; }
