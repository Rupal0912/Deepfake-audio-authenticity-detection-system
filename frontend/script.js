// async function predict() {
//   const fileInput = document.getElementById("audioFile");
//   const resultText = document.getElementById("result");
//   const audioPlayer = document.getElementById("audioPlayer");

//   if (fileInput.files.length === 0) {
//     resultText.innerText = "Please upload a WAV file.";
//     return;
//   }

//   const formData = new FormData();
//   formData.append("file", fileInput.files[0]);

//   resultText.innerText = "Processing...";

//   try {
//     const response = await fetch("/predict", {
//       method: "POST",
//       body: formData,
//     });
//     document.getElementById("audioFile").addEventListener("change", function () {
//   const file = this.files[0];
//   if (file) {
//     audioPlayer.src = URL.createObjectURL(file);
//     audioPlayer.load();
//   }
// });

//     const data = await response.json();
//     resultText.innerText = `Prediction: ${data.prediction.toUpperCase()}`;
//   } catch (error) {
//     resultText.innerText = "Error occurred.";
//   }
// }
const audioInput = document.getElementById("audioFile");
const audioPlayer = document.getElementById("audioPlayer");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");

audioInput.addEventListener("change", () => {
  const file = audioInput.files[0];
  if (file) {
    audioPlayer.src = URL.createObjectURL(file);
    audioPlayer.load();
  }
});

async function predict() {
  if (!audioInput.files.length) {
    alert("Please upload a WAV file.");
    return;
  }

  predictionText.innerText = "Analyzing...";
  confidenceText.innerText = "";

  const formData = new FormData();
  formData.append("file", audioInput.files[0]);

  const response = await fetch("/predict", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  predictionText.innerText =
    data.prediction === "real"
      ? "🟢 Real Human Audio"
      : "🔴 AI-Generated Audio";

  confidenceText.innerText =
    `Confidence → Real: ${data.confidence.real}% | Fake: ${data.confidence.fake}%`;
}

