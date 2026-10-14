// Upload page behaviour: drag & drop, preview, client-side checks, real POST to /api/analyze.
(function () {
  "use strict";
  const zone = document.getElementById("dropzone");
  if (!zone) return;

  const $ = (id) => document.getElementById(id);
  const input = $("file-input"), btn = $("analyze-btn"), change = $("change-btn");
  const empty = $("empty-state"), prev = $("preview-state"), img = $("preview");
  const meta = $("file-meta"), errBox = $("error-box"), status = $("status-line");
  const spinner = $("spinner"), label = $("analyze-label");
  const maxBytes = parseFloat(zone.dataset.maxMb) * 1024 * 1024;
  const okTypes = ["image/jpeg", "image/png"];
  const modelMissing = btn.dataset.modelMissing === "1";
  let file = null;

  function showError(msg) { errBox.textContent = msg; errBox.classList.remove("d-none"); }
  function clearError() { errBox.classList.add("d-none"); errBox.textContent = ""; }

  function reset() {
    file = null; input.value = ""; clearError(); status.textContent = "";
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    img.removeAttribute("src");
    prev.classList.add("d-none"); empty.classList.remove("d-none");
    change.classList.add("d-none"); btn.disabled = true;
  }

  function setFile(f) {
    clearError();
    if (!f) return;
    if (!okTypes.includes(f.type) || !/\.(jpe?g|png)$/i.test(f.name)) {
      showError("Unsupported file type. Please choose a JPG, JPEG or PNG image."); return;
    }
    if (f.size === 0) { showError("The selected file is empty."); return; }
    if (f.size > maxBytes) { showError("The file is too large. Maximum size is " + zone.dataset.maxMb + " MB."); return; }
    file = f;
    if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    img.src = URL.createObjectURL(f);
    meta.textContent = f.name + " · " + (f.size / 1024).toFixed(0) + " KB";
    empty.classList.add("d-none"); prev.classList.remove("d-none");
    change.classList.remove("d-none"); btn.disabled = modelMissing;
    if (modelMissing) status.textContent = "Analysis is disabled because no trained model is loaded.";
  }

  $("browse-btn").addEventListener("click", () => input.click());
  change.addEventListener("click", reset);
  input.addEventListener("change", () => setFile(input.files[0]));
  ["dragenter", "dragover"].forEach((e) => zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((e) => zone.addEventListener(e, (ev) => { ev.preventDefault(); zone.classList.remove("dragover"); }));
  zone.addEventListener("drop", (ev) => setFile(ev.dataTransfer.files[0]));

  btn.addEventListener("click", async () => {
    if (!file) return;
    clearError();
    btn.disabled = true; change.disabled = true; spinner.classList.remove("d-none");
    label.textContent = "Analyzing…"; status.textContent = "Analyzing your image. This can take a few seconds.";
    const fd = new FormData(); fd.append("image", file);
    try {
      const res = await fetch(zone.dataset.api, { method: "POST", body: fd, credentials: "same-origin" });
      let data = null;
      try { data = await res.json(); } catch (_) { /* non-JSON error */ }
      if (!res.ok) throw new Error((data && data.error) || "The analysis failed (HTTP " + res.status + ").");
      window.location.assign(data.results_url);
    } catch (e) {
      showError(e.message || "Network error. Is the server running?");
      status.textContent = "";
      btn.disabled = false; change.disabled = false; spinner.classList.add("d-none"); label.textContent = "Analyze image";
    }
  });
})();
