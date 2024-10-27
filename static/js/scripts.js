/*!
 * Start Bootstrap - SB Admin v7.0.7 (https://startbootstrap.com/template/sb-admin)
 * Copyright 2013-2023 Start Bootstrap
 * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
 */

// Scripts

window.addEventListener('DOMContentLoaded', event => {
    // Toggle the side navigation
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        // Uncomment Below to persist sidebar toggle between refreshes
        // if (localStorage.getItem('sb|sidebar-toggle') === 'true') {
        //     document.body.classList.toggle('sb-sidenav-toggled');
        // }
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }
});

// Data handling function to read dataset details
async function read_data(event, url) {
    event.preventDefault();

    const dataset_id = document.getElementById("dataset_id").value;
    const dataset_name = document.getElementById("dataset_name").value;
    const dataset_date = document.getElementById("dataset_date").value;
    const dataset_model = document.getElementById("dataset_model").value;
    const dataset_url = document.getElementById("dataset_url").value;
    const dataset_username = document.getElementById("dataset_username").value;
    const dataset_password = document.getElementById("dataset_password").value;
    const dataset_path = document.getElementById("dataset_path").value;
    const dataset_format = document.getElementById("dataset_format").value;
    const dataset_description = document.getElementById("dataset_description").value;
    const dataset_metadata = document.getElementById("dataset_metadata").value;
    const dataset_schema = document.getElementById("dataset_schema").value;

    const data = {
        "data": [
            [dataset_id, dataset_name, dataset_date, dataset_model, dataset_url, dataset_username, dataset_password, dataset_path, dataset_format, dataset_description, dataset_metadata, dataset_schema],
        ],
        "columns": ["dataset_id", "dataset_name", "dataset_date", "dataset_model", "dataset_url", "dataset_username", "dataset_password", "dataset_path", "dataset_format", "dataset_description", "dataset_metadata", "dataset_schema"]
    };

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const resultsObj = await response.json();
        console.log(resultsObj);

        if (response.ok) {
            document.getElementById("successMessage").classList.remove("d-none");
            setTimeout(() => { window.location.href = "/catalog"; }, 2000);
        } else {
            console.error('Failed to get response:', response);
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

// Fetch search results and display them in the card format
async function query_search(event, url) {
    event.preventDefault();

    const queryText = document.getElementById("query_text").value;
    const data = { "query_text": queryText };

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error("Failed to fetch results");
        }

        const datasets = await response.json();
        displayResults(datasets);
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

// Display search results as cards with two sections
function displayResults(datasets) {
    const resultsContainer = document.getElementById("resultsContainer");
    resultsContainer.innerHTML = ""; // Clear previous results

    datasets.forEach(dataset => {
        const card = `
            <div class="col-md-6 col-lg-4 mb-4">
                <div class="card" style="background-color: #e6e6fa;">
                    <div class="card-header">${dataset.TableName}</div>
                    <div class="card-body">
                        <div style="border-bottom: 1px solid #ccc; padding-bottom: 10px;">
                            <p><strong>Rows:</strong> ${dataset.Rows || "N/A"}</p>
                            <p><strong>Columns:</strong> ${dataset.Columns || "N/A"}</p>
                            <p><strong>Type:</strong> ${dataset.Type || "N/A"}</p>
                        </div>
                        <div style="margin-top: 10px;">
                            <p><strong>Matching Cell:</strong> ${dataset.CellValue || "N/A"}</p>
                            <p><strong>Column:</strong> ${dataset.CellValue_Column || "N/A"}</p>
                            <p><strong>Similarity Score:</strong> ${dataset.SimilarityScores.toFixed(2)}</p>
                        </div>
                        <div class="d-flex justify-content-between mt-3">
                            <button class="btn btn-outline-primary btn-sm">View</button>
                            <button class="btn btn-outline-primary btn-sm">Download</button>
                        </div>
                    </div>
                </div>
            </div>`;
        resultsContainer.innerHTML += card;
    });
}

// Function to show similar datasets (Placeholder)
function showSimilar(datasetName) {
    alert(`Showing similar datasets for: ${datasetName}`);
    // Implement the function to fetch and display similar datasets
}

// Function to download dataset
function downloadDataset(datasetName) {
    window.location.href = `/download/${datasetName}`;
}

// Function to view dataset content (Placeholder)
function viewDataset(datasetName) {
    alert(`Viewing content for: ${datasetName}`);
    // Implement the function to fetch and display dataset content
}

// Event listener for form submission on page load
document.addEventListener("DOMContentLoaded", function () {
    const form = document.querySelector('form');
    form.addEventListener('submit', query_search);

    // Commented out old table-based search functionality
    /*
    const searchInput = document.getElementById('searchInput');
    const tableBody = document.getElementById('datatablesSimple1').getElementsByTagName('tbody')[0];

    searchInput.addEventListener('input', function () {
        const searchTerm = searchInput.value.toLowerCase();

        for (let i = 0; i < tableBody.rows.length; i++) {
            const row = tableBody.rows[i];
            const rowData = row.textContent.toLowerCase();

            if (rowData.includes(searchTerm)) {
                row.style.display = 'table-row';
            } else {
                row.style.display = 'none';
            }
        }
    });
    */
});

// ########################################################################################################################
// Upload new dataset (CSV)

// Show the confirmation modal
function showConfirmationModal() {
    var myModal = new bootstrap.Modal(document.getElementById('confirmationModal'));
    myModal.show();
}

// Add an event listener to the CSV upload form
const csvUploadForm = document.getElementById("csvUploadForm");
csvUploadForm.addEventListener("submit", function (event) {
    event.preventDefault();  // Prevent default form submission

    // Show the loading spinner
    loadingSpinner.style.display = "block";

    const fileInput = csvUploadForm.querySelector('input[type="file"]');
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Send the file using fetch
    fetch("/csv/", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.message === "CSV file uploaded successfully") {
            showConfirmationModal();
        }
    })
    .catch(error => {
        console.error('Error uploading CSV file:', error);
    })
    .finally(() => {
        loadingSpinner.style.display = "none";
    });
});
