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
  // const processBtn = document.querySelector(".process-btn");
  // if (processBtn) {
  //   processBtn.addEventListener("click", function () {
  //     fetch("/process_data")
  //       .then((response) => response.json())
  //       .then((data) => {
  //         if (data.error) {
  //           console.log("Error:", data.error);
  //           alert(data.error);
  //         } else {
  //           console.log("Processing complete:", data);
  //           document.getElementById("training-accuracy").textContent =
  //             data.training_accuracy.toFixed(2) + "%";
  //           document.getElementById("testing-accuracy").textContent =
  //             data.testing_accuracy.toFixed(2) + "%";

  //           // Redirect to results page after processing
  //           window.location.href = "/hasil_klasifikasi";
  //         }
  //       })
  //       .catch(() => {
  //         alert("An error occurred during processing.");
  //       });
  //   });
  // }
});
