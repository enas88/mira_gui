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

