// ---------------- SHOW FILE NAME ----------------
function showFileName() {
    let f = document.getElementById("audioFile").files[0];
    document.getElementById("fileName").innerText = f ? f.name : "Choose Audio File...";
}


// ---------------- CLEAR FUNCTIONS ----------------
function clearText() {
    document.getElementById("textResult").style.display = "none";
}

function clearAudio() {
    document.getElementById("audioResult").style.display = "none";
    document.getElementById("confidenceBar").style.display = "none";
}


// ---------------- TEXT ANALYSIS ----------------
function analyzeText() {
    let text = document.getElementById("textInput").value;

    if (!text.trim()) return alert("Please enter text!");

    let form = new FormData();
    form.append("text", text);

    fetch("/predict_text", { method: "POST", body: form })
    .then(r => r.json())
    .then(data => {
        
        let box = document.getElementById("textResult");
        box.style.display = "block";

        box.innerHTML = `
        🔍 <b>${data.result}</b><br><br>
        <b>Original:</b><br>${data.original}<br><br>
        <b>Corrected:</b><br>${data.corrected}
        `;
    });
}


// ---------------- AUDIO ANALYSIS ----------------
function uploadAudio() {
    let file = document.getElementById("audioFile").files[0];
    if (!file) return alert("Choose an audio file!");

    let form = new FormData();
    form.append("audio", file);

    fetch("/predict_audio", { method: "POST", body: form })
    .then(r => r.json())
    .then(data => {
        
        let box = document.getElementById("audioResult");
        box.style.display = "block";

        box.innerHTML = `
        🎧 <b>${data.result}</b><br>
        File: ${data.file}<br>
        Confidence: ${data.confidence ? data.confidence.toFixed(2) + "%" : "N/A"}
        `;

        if (data.confidence) {
            document.getElementById("confidenceBar").style.display = "block";
            document.getElementById("confidenceFill").style.width = data.confidence + "%";
        }
    });
}
