// Área de envio de arquivo: mostra o arquivo escolhido (com pré-visualização local), valida tamanho
// e formato antes do envio e deixa trocar ou remover. Sem dependências.
(function () {
  function formatSize(bytes) {
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1).replace(".", ",") + " MB";
    return Math.max(1, Math.round(bytes / 1024)) + " KB";
  }

  function init(root) {
    var input = root.querySelector(".drop-input");
    var zone = root.querySelector(".drop-zone");
    var error = root.querySelector(".drop-error");
    var nameEl = root.querySelector("[data-name]");
    var sizeEl = root.querySelector("[data-size]");
    var preview = root.querySelector("[data-preview]");
    var kind = root.getAttribute("data-kind");
    var maxBytes = parseFloat(root.getAttribute("data-max-mb")) * 1048576;
    var exts = (root.getAttribute("data-exts") || "").split(",").filter(Boolean);
    var objectUrl = null;

    function clearPreview() {
      if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
      preview.innerHTML = "";
    }

    function reset(message) {
      input.value = "";
      clearPreview();
      root.classList.remove("has-file");
      error.hidden = !message;
      error.textContent = message || "";
    }

    function problem(file) {
      var ext = "." + file.name.split(".").pop().toLowerCase();
      var typeOk = file.type && file.type.indexOf(kind + "/") === 0;
      if (!typeOk && exts.indexOf(ext) === -1) {
        return "Formato não suportado. Use " + exts.join(", ").toUpperCase().replace(/\./g, "") + ".";
      }
      if (file.size > maxBytes) {
        return "Arquivo de " + formatSize(file.size) + ": o limite é " + formatSize(maxBytes) + ".";
      }
      if (file.size === 0) return "O arquivo está vazio.";
      return null;
    }

    function show() {
      var file = input.files && input.files[0];
      clearPreview();
      if (!file) { reset(); return; }
      var issue = problem(file);
      if (issue) { reset(issue); return; }
      error.hidden = true;
      nameEl.textContent = file.name;
      sizeEl.textContent = formatSize(file.size);
      objectUrl = URL.createObjectURL(file);
      var media = document.createElement(kind === "video" ? "video" : "audio");
      media.src = objectUrl;
      media.controls = true;
      media.preload = "metadata";
      if (kind === "video") media.muted = true;
      media.setAttribute("aria-label", "Pré-visualização de " + file.name);
      preview.appendChild(media);
      root.classList.add("has-file");
    }

    input.addEventListener("change", show);
    ["dragenter", "dragover"].forEach(function (ev) {
      zone.addEventListener(ev, function () { root.classList.add("over"); });
    });
    ["dragleave", "drop", "dragend"].forEach(function (ev) {
      zone.addEventListener(ev, function () { root.classList.remove("over"); });
    });
    root.querySelector("[data-change]").addEventListener("click", function () { input.click(); });
    root.querySelector("[data-clear]").addEventListener("click", function () { reset(); input.focus(); });
    if (input.files && input.files.length) show();  // volta com arquivo (ex.: botão Voltar)
  }

  function boot() { document.querySelectorAll("[data-drop]").forEach(init); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
