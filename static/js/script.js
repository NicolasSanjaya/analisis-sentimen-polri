// static/js/script.js
document.addEventListener("DOMContentLoaded", function () {
  // File input styling and validation
  const fileInput = document.getElementById("file");
  if (fileInput) {
    fileInput.addEventListener("change", function (e) {
      const fileName = e.target.files[0]?.name;
      if (fileName) {
        // Check if file is CSV
        const fileExt = fileName.split(".").pop().toLowerCase();
        if (fileExt !== "csv") {
          alert("Please upload a CSV file only.");
          fileInput.value = "";
          return;
        }
      }
    });
  }

  // Process button functionality
  const processBtn = document.querySelector(".process-btn");
  if (processBtn) {
    processBtn.addEventListener("click", function () {
      // In a real application, this would submit the processing options
      // For now, just show an alert
      alert("Processing will be performed with the selected options.");
    });
  }
});
