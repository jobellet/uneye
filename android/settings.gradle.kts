// Android app for offline manual saccade labeling (plan phase 2+).
// The C++ (cpp/ + android/offline_session + android/bench) is built by the
// top-level android/CMakeLists.txt through externalNativeBuild below.
pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "uneye-offline-labeler"
include(":app")
