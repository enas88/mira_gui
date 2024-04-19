/*!
    * Start Bootstrap - SB Admin v7.0.7 (https://startbootstrap.com/template/sb-admin)
    * Copyright 2013-2023 Start Bootstrap
    * Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-sb-admin/blob/master/LICENSE)
    */
    // 
// Scripts
// 

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

// document.addEventListener("DOMContentLoaded", function () {
async function read_data(event, url) {
    // console.log('URL:', "/exhaustive_search/");
    // console.log('Headers:', { "Content-Type": "application/json" });

    event.preventDefault();
    const dataset_id = document.getElementById("dataset_id").value;
    const dataset_name = document.getElementById("dataset_name").value;
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
            [dataset_id, dataset_name, dataset_model, dataset_url, dataset_username, dataset_password, dataset_path, dataset_format, dataset_description, dataset_metadata, dataset_schema],
          ],
        "columns": ["dataset_id", "dataset_name", "dataset_model", "dataset_url", "dataset_username", "dataset_password", "dataset_path", "dataset_format", "dataset_description", "dataset_metadata", "dataset_schema"]
    };
    console.log(data);
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
            // Redirect to the home page
            document.getElementById("successMessage").classList.remove("d-none");

            setTimeout(() => {
                window.location.href = "/catalog";
            }, 2000); 
            
        } else {
            console.error('Failed to get response:', response);
        }

    } catch (error) {
        console.error('Error fetching results:', error);
    }
}

async function query_search(event, url) {
    // console.log('URL:', "/exhaustive_search/");
    // console.log('Headers:', { "Content-Type": "application/json" });

    event.preventDefault();
    const query_text = document.getElementById("query_text").value;

    const data = {
        "query_text": query_text,
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
        
        // Get reference to the table body
        var tableBody = document.getElementById('datatablesSimple1').getElementsByTagName('tbody')[0];
        
        // Clear existing rows
        tableBody.innerHTML = "";
        
        // Populate the table
        resultsObj.forEach((result, index) => {
            var row = tableBody.insertRow(index);

            var cell1 = row.insertCell(0);
            var cell2 = row.insertCell(1);
            var cell3 = row.insertCell(2);
            var cell4 = row.insertCell(3);

            // Access the properties from each result object
            cell1.appendChild(document.createTextNode(result.TableName));
            cell2.appendChild(document.createTextNode(result.CellValue));
            cell3.appendChild(document.createTextNode(result.CellValue_Column));
            cell4.appendChild(document.createTextNode(result.SimilaritiyScores));
        });

        const datatablesSimple = document.getElementById('datatablesSimple1');
        if (datatablesSimple) {
            new simpleDatatables.DataTable(datatablesSimple);
        }

        document.getElementById('datatablesSimple1').style.display = 'table';

    } catch (error) {
        console.error('Error fetching results:', error);
    }
}
document.addEventListener("DOMContentLoaded", function () {
    // // Attach the event listener to the form
    const form = document.querySelector('form');
    form.addEventListener('submit', query_search);


    // Get reference to the search input and table body
    const searchInput = document.getElementById('searchInput');
    const tableBody = document.getElementById('datatablesSimple1').getElementsByTagName('tbody')[0];

    // Add an event listener to the search input
    searchInput.addEventListener('input', function () {
        const searchTerm = searchInput.value.toLowerCase();

        // Iterate over the rows and show/hide based on the search term
        for (let i = 0; i < tableBody.rows.length; i++) {
            const row = tableBody.rows[i];
            const rowData = row.textContent.toLowerCase();

            // If the search term is found in the row data, show the row; otherwise, hide it
            if (rowData.includes(searchTerm)) {
                row.style.display = 'table-row';
            } else {
                row.style.display = 'none';
            }
        }


        
    });




});

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