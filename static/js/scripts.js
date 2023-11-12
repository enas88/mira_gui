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

async function query_search(event) {
    console.log('URL:', "/exhaustive_search/");
    console.log('Headers:', { "Content-Type": "application/json" });

    event.preventDefault();
    const query_text = document.getElementById("query_text").value;

    const data = {
        "query_text": query_text,
    };

    try {
        const response = await fetch("/exhaustive_search/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const resultsObj = await response.json();
        console.log(resultsObj)
        // Extract data from the object
        var tableNames = resultsObj.TableName;
        var cellValues = resultsObj.CellValue;
        var cellValueColumns = resultsObj.CellValue_Column;
        var similarityScores = resultsObj.SimilaritiyScores;

        console.log('tableNames:', tableNames);
        console.log('cellValues:', cellValues);
        console.log('cellValueColumns:', cellValueColumns);
        console.log('similarityScores:', similarityScores);

        // Get reference to the table body
        var tableBody = document.getElementById('datatablesSimple').getElementsByTagName('tbody')[0];

        // Clear existing rows
        tableBody.innerHTML = "";
        
        // Populate the table
        for (var i = 0; i < tableNames.length; i++) {
            var row = tableBody.insertRow(i);
        
            var cell1 = row.insertCell(0);
            var cell2 = row.insertCell(1);
            var cell3 = row.insertCell(2);
            var cell4 = row.insertCell(3);
        
            // Access the properties from the nested objects
            cell1.appendChild(document.createTextNode(tableNames[i]));
            cell2.appendChild(document.createTextNode(cellValues[i]));
            cell3.appendChild(document.createTextNode(cellValueColumns[i]));
            cell4.appendChild(document.createTextNode(similarityScores[i]));
        }
    } catch (error) {
        console.error('Error fetching results:', error);
    }
}
document.addEventListener("DOMContentLoaded", function () {
    // // Attach the event listener to the form
    const form = document.querySelector('form');
    form.addEventListener('submit', query_search);
});


