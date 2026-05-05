import { speak } from "@/utils/tts";
import axios from "axios";
import { CameraView, useCameraPermissions } from "expo-camera";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { GestureDetector, Gesture } from "react-native-gesture-handler";

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const [statusMessage, setStatusMessage] = useState("READY");

  const cameraRef = useRef<CameraView>(null);
  const lastCaptionRef = useRef("");
  const isProcessingRef = useRef(false);

  const API_URL = process.env.EXPO_PUBLIC_API_URL;
  const BACKEND_URL = `${API_URL}/analyse-frame`;

  const captureAndSend = useCallback(async () => {
    if (!cameraRef.current || isProcessingRef.current) return;

    if (!API_URL) {
      setStatusMessage("API URL not configured");
      console.warn("EXPO_PUBLIC_API_URL is not set. Please define it in your .env file.");
      return;
    }

    isProcessingRef.current = true;
    setStatusMessage("ANALYSING...");

    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.3,
          base64: false,
        });

        if (!photo?.uri) {
          setStatusMessage("Failed to capture image");
          return;
        }

        const formData = new FormData();
        formData.append("file", {
          uri: photo?.uri,
          name: "frame.jpg",
          type: "image/jpeg",
        } as any);

        console.log("Sending to Backend...");
        const response = await axios.post(BACKEND_URL, formData, {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 10000,
        });
        

        const message = response.data.assistive_message || response.data.caption || "No caption generated.";
        console.log("Assistive message:", message);
        console.log("Image file:", response.data.annotated_image_url);
        console.log("Latency:", response.data.latency_seconds);

        await handleCaption(message);

        if (response.data.latency_seconds) {
          setStatusMessage(`DONE | ${response.data.latency_seconds.toFixed(2)}s`);
        } else {
          setStatusMessage("DONE");
        }
        
      } catch (error: any) {
        console.error("Communication error:", error.message);
        setStatusMessage("BACKEND ERROR");
      } finally {
        isProcessingRef.current = false;
      }
    }
  }, [BACKEND_URL, API_URL]);

    useEffect(() => {
    let isActive: boolean = true;

    const loop = async () => {
      while (isActive && isRecording) {
        await captureAndSend();
        await new Promise((r) => setTimeout(r, 800));

      }
    };

    if (isRecording) {
      loop();
    }

    return () => {
      isActive = false;
    }
  }, [isRecording, captureAndSend]);

  const handleCaption = async (caption: string) => {
    if (!caption || caption === lastCaptionRef.current) return;
      lastCaptionRef.current = caption;

      await speak(caption);
  }

  const tap = Gesture.Tap()
    .runOnJS(true)
    .onStart(() => {
      setIsRecording(!isRecording);
    })

  if (!permission) return <View style={styles.container} />;
  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <View style={styles.permissionWrapper}>
          <Text style={styles.instructionText}>
            ThirdEye needs camera access.
          </Text>
          <TouchableOpacity
            onPress={requestPermission}
            style={styles.permissionButton}
          >
            <Text style={styles.buttonTextSmall}>Allow Camera</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  return (
    <GestureDetector gesture={tap}>
      <View style={styles.container}>
        {/* FIX: CameraView is now self-closing */}
        <CameraView ref={cameraRef} style={styles.camera} facing="back" />

        {/* FIX: UI is now a sibling overlay using absoluteFillObject */}
        <View
          style={[StyleSheet.absoluteFillObject, styles.overlay]}
          pointerEvents="box-none"
        >
          <View style={styles.header}>
            <Text style={styles.appName}>THIRDEYE</Text>
          </View>

          <View style={[styles.statusContainer,
            isRecording ? styles.active : styles.inactive,
          ]}>
            <View
              style={[
                styles.dot,
                { backgroundColor: isRecording ? "#ff4444" : "#555" },
              ]}
            />
            <Text style={styles.statusText}>
              {statusMessage}
            </Text>
          </View>
        </View>
      </View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  camera: { flex: 1 },
  header: { position: "absolute", top: 70 },
  appName: {
    color: "white",
    fontSize: 14,
    fontWeight: "800",
    letterSpacing: 4,
    opacity: 0.7,
  },
  permissionWrapper: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 40,
  },
  instructionText: {
    color: "white",
    fontSize: 20,
    textAlign: "center",
    marginBottom: 40,
  },
  permissionButton: {
    backgroundColor: "#2563eb",
    paddingVertical: 18,
    paddingHorizontal: 40,
    borderRadius: 40,
  },
  overlay: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "rgba(0,0,0,0.15)",
  },
  mainButton: {
    width: 240,
    height: 240,
    borderRadius: 120,
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 10,
    borderColor: "white",
  },
  active: { backgroundColor: "rgba(220, 38, 38, 0.85)" },
  inactive: { backgroundColor: "rgba(16, 185, 129, 0.85)" },
  buttonText: { color: "white", fontSize: 40, fontWeight: "900" },
  buttonTextSmall: { color: "white", fontSize: 18, fontWeight: "bold" },
  statusContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 40,
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 25,
    opacity: 0.8
  },
  dot: { width: 10, height: 10, borderRadius: 5, marginRight: 10 },
  statusText: {
    color: "white",
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 1,
  },
});
