async function predict() {
    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please upload an audio file.");
        return;
    }

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("result").classList.add("hidden");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        document.getElementById("loading").classList.add("hidden");
        document.getElementById("result").classList.remove("hidden");

        if (data.error) {
            document.getElementById("prediction").innerText = "Error: " + data.error;
            return;
        }

        const pred = data.prediction;
        const probs = data.probability;

        document.getElementById("prediction").innerHTML =
            pred === 1
                ? "🧪 <b>Fake Audio Detected</b>"
                : "🟢 <b>Real Human Audio</b>";

        document.getElementById("probabilities").innerHTML =
            `Confidence → Real: ${(probs[0] * 100).toFixed(2)}% |
             Fake: ${(probs[1] * 100).toFixed(2)}%`;

    } catch (e) {
        document.getElementById("loading").classList.add("hidden");
        alert("Request failed: " + e);
    }
}
