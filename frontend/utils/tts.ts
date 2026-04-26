import * as Speech from "expo-speech";

export const speak = (text: string) => {
  if (!text) return;

  Speech.stop();

  Speech.speak(text, {
    language: "en-GB",
    rate: 0.9,
    pitch: 1.0,
  });
};

export const stopSpeaking = () => {
  Speech.stop();
};