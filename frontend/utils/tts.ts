import * as Speech from "expo-speech";

export const speak = (text: string): Promise<void> => {
  if (!text) return Promise.resolve()

  
  return new Promise(async (resolve) => {
    await Speech.stop();

    Speech.speak(text, {
      language: "en-GB",
      rate: 0.9,
      pitch: 1,
      onDone: resolve,
      onStopped: resolve
    }
    );
  })
}

export const stopSpeaking = () => {
  Speech.stop();
}