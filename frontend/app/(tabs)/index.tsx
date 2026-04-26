import { speak } from "@/utils/tts";
import axios from "axios";
import { CameraView, useCameraPermissions } from "expo-camera";
import React, { useEffect, useRef, useState } from "react";
import { StyleSheet, Text, TouchableOpacity, View } from "react-native";

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [isRecording, setIsRecording] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  const lastCaptionRef = useRef("");

  const LAPTOP_IP = "10.77.111.156";
  const BACKEND_URL = `http://${LAPTOP_IP}:8000/analyse-frame`;

  useEffect(() => {
    let interval: any;
    if (isRecording) {
      captureAndSend();
      interval = setInterval(captureAndSend, 5000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  const captureAndSend = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.3,
          base64: false,
        });

        const formData = new FormData();
        formData.append("file", {
          uri: photo?.uri,
          name: "frame.jpg",
          type: "image/jpeg",
        } as any);

        console.log("Sending to Backend...");
        const response = await axios.post(BACKEND_URL, formData, {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 4000,
        });

        console.log("AI Caption:", response.data.caption);
        handleCaption(response.data.caption);
        
      } catch (error: any) {
        console.error("Communication error:", error.message);
      }
    }
  };

  const handleCaption = (caption: string) => {
    if (!caption || caption === lastCaptionRef.current) return;
      lastCaptionRef.current = caption;
      speak(caption);
  }

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

        <TouchableOpacity
          activeOpacity={0.8}
          style={[
            styles.mainButton,
            isRecording ? styles.active : styles.inactive,
          ]}
          onPress={() => setIsRecording(!isRecording)}
        >
          <Text style={styles.buttonText}>
            {isRecording ? "STOP" : "START"}
          </Text>
        </TouchableOpacity>

        <View style={styles.statusContainer}>
          <View
            style={[
              styles.dot,
              { backgroundColor: isRecording ? "#ff4444" : "#555" },
            ]}
          />
          <Text style={styles.statusText}>
            {isRecording ? "ANALYZING" : "READY"}
          </Text>
        </View>
      </View>
    </View>
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
  },
  dot: { width: 10, height: 10, borderRadius: 5, marginRight: 10 },
  statusText: {
    color: "white",
    fontSize: 14,
    fontWeight: "700",
    letterSpacing: 1,
  },
});
