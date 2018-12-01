function buildMetadata() {
  // @TODO: Complete the following function that builds the metadata panel
  // Use `d3.json` to fetch the metadata for a sample
  dr_no = d3.select("#selDataset").property("value");
  d3.json("/crime_sites/"+dr_no).then(function(metaData) {

    function pad(n) {
      return (n < 1000000) ? ("0" + n) : n;
    };
    
    var site_id = pad(metaData.dr_no);
    console.log(dr_no);


      // Use d3 to select the panel with id of `#sample-metadata` and id of `#gauge`
      var metaPanel = d3.select("#sample-metadata");

      // Use `.html("") to clear any existing metadata and gauge
      metaPanel.html("");

    // Use `Object.entries` to add each key and value pair to the panel
    // Hint: Inside the loop, you will need to use d3 to append new
    // tags for each key-value in the metadata.  
      // Object.entries(metaData).metaPanel.insert(`<tr>Name: ${name}`);
      d3.select("#sample-metadata")
        .html(`<h3>${metaData.crm_cd_desc}<h3>
               <h5>Area: ${metaData.area_name}<br>
               Location: ${metaData.location}<br>
               Cross Street: ${metaData.cross_street}<br>
               Date: ${metaData.date_occ.slice(0,16)}<br>
               Hour: ${metaData.hour_occ}<br>
               Weapon Used: ${metaData.weapon_desc}<br>
               `);

      d3.select("#sitemap_title").html(`Sattelite View: ${metaData.location}`);
      d3.select("#tableTitle").html(`LA Crime Detail`);
   
      });
}

function buildMap() {
  idname = d3.select("#selDataset").property("value");
  d3.json("/crime_sites/"+dr_no).then(function(metaData) {
    var latitude = metaData.latitude;
    var longitude = metaData.longitude;

   document.getElementById('sitemap').innerHTML = '<div id="maptwo"></div>';
    
    // Add satelite tile layer
    var satelliteMap = new L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
      attribution: "Map data &copy; <a href='https://www.openstreetmap.org/'>OpenStreetMap</a> contributors, <a href='https://creativecommons.org/licenses/by-sa/2.0/'>CC-BY-SA</a>, Imagery © <a href='https://www.mapbox.com/'>Mapbox</a>",
      maxZoom: 18,
      id: "mapbox.satellite",
      accessToken: API_KEY
    });

    // Create map object
    var siteMap = new L.map("maptwo", {
      center: [latitude, longitude],
      zoom: 18,
      layers: [satelliteMap]
    
    });

  });
}


function init() {

  // Grab a reference to the dropdown select element
  var selector = d3.select("#selDataset");
  
  // Create list of sample id's and use it to populate the select options
  d3.json("/crime_sites").then(function(crime_sites) {
    crime_sites.forEach(function(data) {
      data.dr_no = +data.dr_no;
      data.area_name = data.area_name;
      data.location = data.location;
      data.cross_street = data.cross_street;
      data.crm_cd_desc = data.crm_cd_desc;
      data.weapon_desc = data.weapon_desc;
      data.date_occ = data.date_occ;
      data.hour_occ = +data.hour_occ;
      data.latitude = +data.latitude;
      data.longitude = +data.longitude;
      data.crm_cd = +data.crm_cd;
  });
  // var sampleId = superfundData.map(d => d.id);
  var sampleIdName = crime_sites.map(d => d.dr_no);
   
    sampleIdName.forEach((dr_no) => {
      selector
        .append("option")
        .text(dr_no)
        .property("value", dr_no);
  });
 
    // Use the first sample from the list to build the initial plots
    const firstSample = sampleIdName[0];
    // buildCharts(firstSample);
    buildMetadata(firstSample);
    buildMap(firstSample);
    console.log(firstSample);
  });  
}

function optionChanged(newSample) {
  // Fetch new data each time a new sample is selected
  // buildCharts(newSample);
  buildMetadata(newSample);
  buildMap(newSample);
  console.log(newSample);
}
// Initialize the dashboard
init();