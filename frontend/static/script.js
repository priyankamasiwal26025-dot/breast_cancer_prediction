async function upload() {
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select an image file first.');
        return;
    }
    const formData = new FormData();
    formData.append('image', file);

    const btn = document.getElementById('btn');
    btn.disabled = true;
    btn.innerText = 'Predicting...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();  // expects JSON!

        document.getElementById('result').style.display = 'block';

        if (response.ok) {
            document.getElementById('out').innerHTML =
                `<strong>${data.label}</strong> (malignant probability: ${(data.malignant_probability * 100).toFixed(2)}%)`;
            document.getElementById('meta').innerText =
                `Raw model output: ${data.malignant_probability}`;
        } else {
            document.getElementById('out').innerText =
                `Error: ${data.error || 'unknown'}`;
            document.getElementById('meta').innerText = '';
        }
    } catch (err) {
        document.getElementById('result').style.display = 'block';
        document.getElementById('out').innerText =
            'Network or server error: ' + String(err);
        document.getElementById('meta').innerText = '';
    } finally {
        btn.disabled = false;
        btn.innerText = 'Predict';
    }
}
