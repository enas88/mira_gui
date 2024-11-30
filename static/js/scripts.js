/*!
 * Start Bootstrap - SB Admin v7.0.7 (https://startbootstrap.com/template/sb-admin)
 * Copyright 2013-2023 Start Bootstrap
 * Licensed under MIT
 */

// Scripts
window.addEventListener('DOMContentLoaded', event => {
    const sidebarToggle = document.body.querySelector('#sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', event => {
            event.preventDefault();
            document.body.classList.toggle('sb-sidenav-toggled');
            localStorage.setItem('sb|sidebar-toggle', document.body.classList.contains('sb-sidenav-toggled'));
        });
    }
});

async function query_search(event, url, displayMode = "cards") {
    event.preventDefault();
    console.log("query_search triggered");

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

        // Display results based on the displayMode (default is "cards")
        if (displayMode === "table") {
            displayResultsInTable(datasets); // Table display for exhaustive or approximate search
        } else {
            paginateResults(datasets); // Card display for optimized search
        }

        // Populate top matching tables if applicable
        populateTopMatchingTables(datasets);

        // Show the "View as Graph" button only if there are results
        const viewGraphButton = document.getElementById("viewGraphButton");
        if (datasets.length > 0) {
            viewGraphButton.style.display = "block";
        } else {
            viewGraphButton.style.display = "none";
        }

    } catch (error) {
        console.error('Error fetching results:', error);
    }
}


// Manages pagination of results
let currentPage = 1;
const cardsPerPage = 4;

function paginateResults(datasets) {
    const totalPages = Math.ceil(datasets.length / cardsPerPage);
    const start = (currentPage - 1) * cardsPerPage;
    const end = start + cardsPerPage;
    displayResults(datasets.slice(start, end)); // Show results for the current page

    const paginationContainer = document.getElementById("paginationContainer");
    paginationContainer.innerHTML = ""; // Clear previous pagination buttons

    const prevButton = document.createElement("button");
    prevButton.innerText = "Previous Page";
    prevButton.classList.add("pagination-button");
    prevButton.onclick = () => {
        if (currentPage > 1) {
            currentPage--;
            paginateResults(datasets);
        }
    };
    prevButton.disabled = currentPage === 1;
    paginationContainer.appendChild(prevButton);

    const nextButton = document.createElement("button");
    nextButton.innerText = "Next Page";
    nextButton.classList.add("pagination-button");
    nextButton.onclick = () => {
        if (currentPage < totalPages) {
            currentPage++;
            paginateResults(datasets);
        }
    };
    nextButton.disabled = currentPage === totalPages;
    paginationContainer.appendChild(nextButton);
}

function displayResults(datasets) {
    const resultsContainer = document.getElementById("resultsContainer");
    resultsContainer.innerHTML = ""; // Clear previous results
    const colors = ["#f8d7da", "#d1ecf1", "#d4edda", "#fff3cd"];

    datasets.forEach(dataset => {
        const color = colors[datasets.indexOf(dataset) % colors.length];
        const card = `
            <div class="col-md-6 col-lg-6 mb-4">
                <div class="card smaller-card" style="background-color: ${color}; display: flex; flex-direction: row;">
                    <div class="card-body" style="flex: 1;">
                        <p><strong>Table Name:</strong> ${dataset.TableName}</p>
                        <p><strong>Rows:</strong> ${dataset.Rows || "N/A"}</p>
                        <p><strong>Columns:</strong> ${dataset.Columns || "N/A"}</p>
                        <p><strong>Type:</strong> ${dataset.Type || "N/A"}</p>
                    </div>
                    <div class="vertical-line"></div>
                    <div class="card-body" style="flex: 1;">
                        <p><strong>Matching Cell:</strong> ${dataset.CellValue || "N/A"}</p>
                        <p><strong>Column:</strong> ${dataset.CellValue_Column || "N/A"}</p>
                        <p><strong>Similarity Score:</strong> ${dataset.SimilarityScores ? dataset.SimilarityScores.toFixed(2) : "N/A"}</p>
                    </div>
                </div>
                <div class="button-container text-center">
                    <button class="btn btn-outline-primary btn-sm" onclick="viewDataset('${dataset.TableName}')">View</button>
                    <a href="/download/${dataset.TableName}" class="btn btn-outline-primary btn-sm" download>Download</a>
                </div>
            </div>
        `;
        resultsContainer.innerHTML += card;
    });
}

function populateTopMatchingTables(datasets) {
    const matchingTablesContainer = document.getElementById("matchingTablesContainer");
    matchingTablesContainer.innerHTML = ''; // Clear previous entries

    datasets.forEach(dataset => {
        const tableItem = document.createElement('div');
        tableItem.classList.add('table-item');
        tableItem.textContent = dataset.TableName;
        tableItem.onclick = () => viewDataset(dataset.TableName);
        matchingTablesContainer.appendChild(tableItem);
    });
}

// Function to display the selected dataset in a modal
async function viewDataset(datasetName) {
    try {
        const response = await fetch(`/get_table/${datasetName}`);
        
        if (!response.ok) {
            throw new Error("Failed to fetch table data");
        }

        const tableData = await response.json();

        // Check if tableData contains the expected structure for either API response
        if (tableData.columns && tableData.data) {
            // Standard structure - most likely from efficient search
            displayTableInModal(tableData);
        } else if (Array.isArray(tableData) && tableData.length > 0) {
            // Adjust if ann_search returns data in a slightly different format
            const transformedData = {
                columns: Object.keys(tableData[0]), // Use keys from the first item as columns
                data: tableData.map(row => Object.values(row))
            };
            displayTableInModal(transformedData);
        }
    } catch (error) {
        console.error('Error fetching table data:', error);
    }
}


// Function to display data inside a modal as a table
function displayTableInModal(tableData) {
    const tableContainer = document.getElementById("tableContainer");
    tableContainer.innerHTML = ""; // Clear previous content

    let tableHTML = `<table class="table table-striped"><thead><tr>`;
    const columns = tableData.columns;
    columns.forEach(column => {
        tableHTML += `<th>${column}</th>`;
    });
    tableHTML += `</tr></thead><tbody>`;
    tableData.data.forEach(row => {
        tableHTML += `<tr>`;
        row.forEach(cell => {
            tableHTML += `<td>${cell}</td>`;
        });
        tableHTML += `</tr>`;
    });
    tableHTML += `</tbody></table>`;

    tableContainer.innerHTML = tableHTML;

    // Show the modal
    const modal = new bootstrap.Modal(document.getElementById('tableViewModal'));
    modal.show();
}


function displayResultsInTable(datasets) {
    const resultsContainer = document.getElementById("resultsContainer");
    resultsContainer.innerHTML = ""; // Clear previous results

    // Create a table
    let tableHTML = `
        <table class="table table-striped">
            <thead>
                <tr>
                    <th>Table Name</th>
                    <th>Matching Cell</th>
                    <th>Column</th>
                    <th>Similarity Score</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>`;

    datasets.forEach(dataset => {
        // Check if SimilarityScores is a number and format it to 2 decimal places, or default to "N/A"
        const similarityScore = typeof dataset.SimilarityScores === 'number'
            ? dataset.SimilarityScores.toFixed(2)
            : "N/A";

        tableHTML += `
            <tr>
                <td>${dataset.TableName || "N/A"}</td>
                <td>${dataset.CellValue || "N/A"}</td>
                <td>${dataset.CellValue_Column || "N/A"}</td>
                <td>${similarityScore}</td>
                <td>
                <button class="btn btn-outline-primary btn-sm" onclick="viewDataset('${dataset.TableName}')">View</button>
                    <a href="/download/${dataset.TableName}" class="btn btn-outline-primary btn-sm" download>Download</a>
                </td>
            </tr>`;
    });

    tableHTML += `</tbody></table>`;
    resultsContainer.innerHTML = tableHTML;
}


// ########################################################################################################################
// Upload new dataset (CSV)


// Show the confirmation modal
function showConfirmationModal() {
    var myModal = new bootstrap.Modal(document.getElementById('confirmationModal'));
    myModal.show();
}


// Add an event listener to the form
const csvUploadForm = document.getElementById("csvUploadForm");
csvUploadForm.addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent the default form submission


    // Show the loading spinner
    loadingSpinner.style.display = "block";

    // Get the file input element
    const fileInput = csvUploadForm.querySelector('input[type="file"]');

    // Create a FormData object to send the file
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    // Send the file using a fetch request
    fetch("/csv/", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        // If calculations are completed successfully, show the confirmation modal
        if (data.message === "CSV file uploaded successfully") {
            showConfirmationModal();
        }
    })
    .catch(error => {
        console.error('Error uploading CSV file:', error);
    })
    .finally(() => {
        // Hide the loading spinner after the fetch request is complete
        loadingSpinner.style.display = "none";
    });

    // Function to show the confirmation modal
    function showConfirmationModal() {
        var myModal = new bootstrap.Modal(document.getElementById('confirmationModal'));
        myModal.show();
    }

});



function applyAnimation(animationType) {
    const logo = document.getElementById("logo");

    // Remove any existing animation classes
    logo.classList.remove("drop", "fade");

    // Trigger reflow to reset animation (allows reapplying the same animation)
    void logo.offsetWidth;

    // Add the selected animation class
    logo.classList.add(animationType);
}
