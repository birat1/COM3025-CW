import * as Speech from "expo-speech";

export const speak = async (text: string) => {
  if (!text) return;

  try {
    await Speech.stop();

    Speech.speak(text, {
      language: "en-GB",
      rate: 0.9,
      pitch: 1.0,
    });
  } catch (error) {
    console.error("Speech error:", error);
  }
};

export const stopSpeaking = () => {
  Speech.stop();
};