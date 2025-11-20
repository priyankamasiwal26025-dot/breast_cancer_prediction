async function upload(){
const f = document.getElementById('file').files[0];
if(!f){ alert('Please select an image file or ensure sample image is accessible.'); return; }
const fd = new FormData();
fd.append('image', f);
const btn = document.getElementById('btn');
btn.disabled = true; btn.innerText = 'Predicting...';


try {
const res = await fetch('/predict', { method:'POST', body: fd });
const j = await res.json();
document.getElementById('result').style.display = 'block';
if(res.ok){
document.getElementById('out').innerHTML = `<strong>${j.label}</strong> (malignant probability: ${(j.malignant_probability*100).toFixed(2)}%)`;
document.getElementById('meta').innerText = `Raw model output: ${j.malignant_probability}`;
} else {
document.getElementById('out').innerText = `Error: ${j.error || 'unknown'}`;
document.getElementById('meta').innerText = '';
}
} catch(err){
document.getElementById('result').style.display = 'block';
document.getElementById('out').innerText = 'Network or server error: ' + String(err);
document.getElementById('meta').innerText = '';
} finally {
btn.disabled = false; btn.innerText = 'Predict';
}
}