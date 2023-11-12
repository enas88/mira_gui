document.addEventListener("DOMContentLoaded", function () {

    function addPerson(event) {
        event.preventDefault();
        const first_name = document.getElementById("first_name").value;
        const last_name = document.getElementById("last_name").value;
        const age = parseInt(document.getElementById("age").value);

        const data = {
            "first_name": first_name,
            "last_name": last_name,
            "age": age
        };

        fetch("/person/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(() => listPersons())
        .catch(error => {
            console.error('Error adding a person:', error);
        });
    }


    function listPersons() {
        // Fetch and display all records
        fetch('/person/')
        .then(response => response.json())
        .then(data => {
            const personTable = document.getElementById('personTable');
            const tbody = personTable.querySelector('tbody');

            // Clear the existing rows
            tbody.innerHTML = '';

            // Loop through the data and create table rows
            data.forEach(person => {
                const row = document.createElement('tr');
                const firstNameCell = document.createElement('td');
                firstNameCell.textContent = person.first_name;
                const lastNameCell = document.createElement('td');
                lastNameCell.textContent = person.last_name;
                const ageCell = document.createElement('td');
                ageCell.textContent = person.age;

                row.appendChild(firstNameCell);
                row.appendChild(lastNameCell);
                row.appendChild(ageCell);

                tbody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
    }


    // Function to fetch and display CSV records
    function listCSVRecords() {
        const csvTableBody = document.getElementById('csvTableBody');
        csvTableBody.innerHTML = '';  // Clear existing rows

        // Fetch and display all CSV records
        fetch('/csv/')
        .then(response => response.json())
        .then(data => {
            data.forEach(csvRecord => {
                const row = document.createElement('tr');
                const nameCell = document.createElement('td');
                nameCell.textContent = csvRecord.name;

                row.appendChild(nameCell);
                csvTableBody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Error fetching CSV records:', error);
        });
    }

    // Initial loading of CSV records
    listCSVRecords();

    // Function to add a new CSV record
    function addCSVRecord(name) {
        const data = { "name": name };

        fetch("/csv/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(() => {
            listCSVRecords();  // Update the table after adding a new record
        })
        .catch(error => {
            console.error('Error adding a CSV record:', error);
        });
    }

    // Add an event listener to the form
    const csvUploadForm = document.getElementById("csvUploadForm");
    csvUploadForm.addEventListener("submit", function (event) {
        event.preventDefault(); // Prevent the default form submission

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
            // Handle the response as needed (e.g., show a success message)
            console.log(data.message);
            listCSVRecords(); // Update the CSV records table
        })
        .catch(error => {
            console.error('Error uploading CSV file:', error);
        });
    });

