package com.rnllama;

import android.os.Build;
import android.util.Log;

import com.facebook.react.bridge.ReactApplicationContext;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Locale;

public class RNLlama {
  public static final String NAME = "RNLlama";
  private static final String TAG = "RNLlama";
  private static boolean libsLoaded = false;

  private final ReactApplicationContext reactContext;

  public RNLlama(ReactApplicationContext reactContext) {
    this.reactContext = reactContext;
  }

  public static synchronized boolean loadNative(ReactApplicationContext context) {
    if (libsLoaded) return true;

    if (Build.SUPPORTED_64_BIT_ABIS.length == 0) {
      Log.w(TAG, "Only 64-bit architectures are supported");
      return false;
    }

    String cpuFeatures = getCpuFeatures();
    boolean hasFp16 = cpuFeatures.contains("fp16") || cpuFeatures.contains("fphp");
    boolean hasDotProd = cpuFeatures.contains("dotprod") || cpuFeatures.contains("asimddp");
    boolean hasI8mm = cpuFeatures.contains("i8mm");

    try {
      boolean jniLoaded = false;
      String loadedLib = "";
      if (isArm64V8a()) {
        if (hasDotProd && hasI8mm) {
          if (tryLoadLibrary("rnllama_jni_v8_2_dotprod_i8mm_opencl")) {
            jniLoaded = true;
            loadedLib = "rnllama_jni_v8_2_dotprod_i8mm_opencl";
          }
        }

        if (!jniLoaded && hasDotProd && hasI8mm) {
          if (tryLoadLibrary("rnllama_jni_v8_2_dotprod_i8mm")) {
            jniLoaded = true;
            loadedLib = "rnllama_jni_v8_2_dotprod_i8mm";
          }
        }

        if (!jniLoaded && hasDotProd) {
          if (tryLoadLibrary("rnllama_jni_v8_2_dotprod")) {
            jniLoaded = true;
            loadedLib = "rnllama_jni_v8_2_dotprod";
          }
        }

        if (!jniLoaded && hasI8mm) {
          if (tryLoadLibrary("rnllama_jni_v8_2_i8mm")) {
            jniLoaded = true;
            loadedLib = "rnllama_jni_v8_2_i8mm";
          }
        }

        if (!jniLoaded && hasFp16) {
          if (tryLoadLibrary("rnllama_jni_v8_2")) {
            jniLoaded = true;
            loadedLib = "rnllama_jni_v8_2";
          }
        }

        if (!jniLoaded) {
          if (tryLoadLibrary("rnllama_jni_v8")) {
            jniLoaded = true;
            loadedLib = "rnllama_jni_v8";
          }
        }
      } else if (isX86_64()) {
        if (tryLoadLibrary("rnllama_jni_x86_64")) {
          jniLoaded = true;
          loadedLib = "rnllama_jni_x86_64";
        }
      } else {
        if (tryLoadLibrary("rnllama_jni")) {
          jniLoaded = true;
          loadedLib = "rnllama_jni";
        }
      }

      if (!jniLoaded) {
        System.loadLibrary("rnllama_jni");
        loadedLib = "rnllama_jni";
      }

      System.loadLibrary("rnllama");
      nativeSetLoadedLibrary(loadedLib);
      libsLoaded = true;
      return true;
    } catch (UnsatisfiedLinkError e) {
      Log.e(TAG, "Failed to load native libraries", e);
      return false;
    }
  }

  private static native void nativeSetLoadedLibrary(String name);

  private static boolean isArm64V8a() {
    for (String abi : Build.SUPPORTED_ABIS) {
      if ("arm64-v8a".equalsIgnoreCase(abi)) return true;
    }
    return false;
  }

  private static boolean isX86_64() {
    for (String abi : Build.SUPPORTED_ABIS) {
      if ("x86_64".equalsIgnoreCase(abi)) return true;
    }
    return false;
  }

  private static String getCpuFeatures() {
    StringBuilder features = new StringBuilder();
    try (BufferedReader br = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
      String line;
      while ((line = br.readLine()) != null) {
        String lower = line.toLowerCase(Locale.ROOT);
        if (lower.startsWith("features") || lower.startsWith("flags")) {
          int idx = lower.indexOf(':');
          if (idx != -1 && idx + 1 < lower.length()) {
            features.append(lower.substring(idx + 1).trim()).append(" ");
          }
        }
      }
    } catch (IOException ignored) {
    }
    return features.toString();
  }

  private static boolean tryLoadLibrary(String libraryName) {
    try {
      System.loadLibrary(libraryName);
      return true;
    } catch (UnsatisfiedLinkError ignored) {
      return false;
    }
  }
}
