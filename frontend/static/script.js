const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");

const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const benignProb = document.getElementById("benignProb");
const malignantProb = document.getElementById("malignantProb");
const recommendText = document.getElementById("recommendText");
const confidenceBar = document.getElementById("confidenceBar");
const title = document.getElementById("result-title");

let selectedImage = null;

// CLICK to open file dialog
document.getElementById("drop-area").onclick = () => imageInput.click();

// SHOW IMAGE PREVIEW
imageInput.onchange = function () {
    selectedImage = this.files[0];
    preview.src = URL.createObjectURL(selectedImage);
};

// RESET button
resetBtn.onclick = () => {
    imageInput.value = "";
    preview.src = "";
    selectedImage = null;

    predictionText.textContent = "--";
    confidenceText.textContent = "--%";
    benignProb.textContent = "--%";
    malignantProb.textContent = "--%";
    recommendText.textContent = "--";
    confidenceBar.style.width = "0%";
    title.textContent = "Prediction will appear here";
};

// ANALYZE button
analyzeBtn.onclick = async function () {
    if (!selectedImage) {
        alert("Please upload an image first!");
        return;
    }

    let formData = new FormData();
    formData.append("image", selectedImage);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    let prob = data.malignant_probability * 100;

    predictionText.textContent = data.label;
    confidenceText.textContent = prob.toFixed(1) + "%";
    malignantProb.textContent = prob.toFixed(1) + "%";
    benignProb.textContent = (100 - prob).toFixed(1) + "%";
    confidenceBar.style.width = prob + "%";

    title.textContent = data.label === "malignant" 
        ? "Malignant Tissue Detected" 
        : "Benign Tissue Detected";

    recommendText.textContent =
        data.label === "malignant"
            ? "The analysis suggests malignant characteristics. Immediate medical evaluation is recommended."
            : "The tissue appears benign. Continue regular screenings as recommended.";
};
