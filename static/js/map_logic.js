// Grabbing LA crime site data.
var crimeSites = [];
var zipStats = [];

d3.json("/crime_sites", function (data) {

  data.forEach(function (sqlitedata) {
    data.crm_cd = + sqlitedata.crm_cd;
    data.crm_cd_desc = sqlitedata.crm_cd_desc;
    data.area_name = sqlitedata.area_name;
    data.location = sqlitedata.location;
    data.cross_street = sqlitedata.cross_street;
    data.date_occ = sqlitedata.date_occ;
    data.hour_occ = +sqlitedata.hour_occ;
    data.latitude = +sqlitedata.latitude;
    data.longitude = +sqlitedata.longitude;
    data.dr_no = +sqlitedata.dr_no;
    crimeSites.push(
      L.circleMarker([data.latitude, data.longitude],
        {
          radius: 5,
          fillColor: getColorCrimeSites(data.crm_cd),
          fillOpacity: .7,
          stroke: true,
          color: "black",
          weight: .5
        }
        //,{ icon: L.BeautifyIcon.icon(options) }
      ).bindPopup(data.crm_cd_desc.bold() + "<br>" + data.location + "<br>" + data.area_name + " Area" + "<br>" + data.date_occ.slice(0,16) + " @ " + data.hour_occ + ":00" + "<br>DivRec#: " + data.dr_no.toString().bold()));
  });

  var zipEmploymentRateChloro;
  var zipIncomeChloro;
  var zipADIChloro;
  var zipDeathsChloro;
  var zipOpioidRxChloro;
  var EmpRateLegend;
  var InLegend;
  var ADILegend;
  var DeathsLegend;
  var OpioidRxLegend;

  d3.json("/crime_stats/get_data", function (data) {
    data.forEach(function (xdata) {
      var xzip = xdata.zip;
      objIndex = zip_boundaries.features.findIndex((obj => obj.properties.name == xzip));
      zip_boundaries.features[objIndex].properties.la_zip_stats = xdata;
    });

    zip_boundaries.features = zip_boundaries.features.filter(function (item) {
      return la_zips.indexOf(item.properties.name) > -1;  // la_zips is from la_zips.js, which contains a list of zips for LA that match to the geojson file for zipcode boundaries
    });

    console.log(la_zips);

    // extract min-max data needed for legends
    var res = createZipChloro("pct_full_time_employed_pop", zip_boundaries, "Percent Employed");
    zipEmploymentRateChloro = res[0];
    EmpRateLegend = res[1];

    res = createZipChloro("household_median_income", zip_boundaries, "Median Income");    
    zipIncomeChloro = res[0];
    InLegend = res[1];

    res = createZipChloro("adi_state_rank", zip_boundaries, "Neighborhood Deprivation");    
    zipADIChloro = res[0];
    ADILegend = res[1];

    res = createZipChloro("total_deaths_per_1000", zip_boundaries, "Deaths per 1,000 Pop");    
    zipDeathsChloro = res[0];
    DeathsLegend = res[1];

    res = createZipChloro("opioid_rx_per_1000", zip_boundaries, "Opioid Rx per 1,000 Pop");    
    zipOpioidRxChloro = res[0];
    OpioidRxLegend = res[1];


    createMap(crimeSites,
      zipEmploymentRateChloro,EmpRateLegend,
      zipIncomeChloro,InLegend,
      zipADIChloro,ADILegend,
      zipDeathsChloro,DeathsLegend,
      zipOpioidRxChloro,OpioidRxLegend);

  });

});

function createZipChloro(prop,sdata,name) {

  var chloro = new L.LayerGroup();

  // All features together
  function XonEachFeature(feature, layer) {
    layer.on({
      //mouseover: highlightFeature,
      //mouseout: resetHighlight,
      //click: zoomToFeature
    });
  }

  // Add style to assign zip color based on employment rate
  function Xstyle(feature) {
    var prop_max = prop+"_max";
    var prop_min = prop+"_min";

    return {
      fillColor: colorScale(feature.properties.la_zip_stats[prop],
        feature.properties.la_zip_stats[prop_min],
        feature.properties.la_zip_stats[prop_max]),
      weight: 2,
      opacity: 0.3,
      color: 'white',
      dashArray: '3',
      //fillOpacity: 1,
      zindex: -1
    };
  }

  geojson = L.geoJson(zip_boundaries, {
      style: Xstyle,
      onEachFeature: XonEachFeature
   }).addTo(chloro);


  var xlegend = L.control({ position: 'bottomleft' });
  xlegend.onAdd = function () {

    var prop_max = prop + "_max";
    var prop_min = prop + "_min";

    // compute grades
    var min = sdata.features[0].properties.la_zip_stats[prop_min];
    var max = sdata.features[0].properties.la_zip_stats[prop_max];

    var bgrades = [0,1,2,3,4,5,6]
    for(i=0;i<bgrades.length;i++) {
      bgrades[i] = Math.round(bgrades[i]/7.0 * (max-min) + min);
    }

    var div = L.DomUtil.create('div', 'info legend'),
      grades = bgrades,
      labels = [];

    div.innerHTML += '<b>'+name+'</b><br>'
    // Loop through intervals to generate labels and colored squares for crime site legend
    for (var i = 0; i < grades.length; i++) {
      div.innerHTML +=
        '<i style="background:' + colorScale(grades[i] + 1,min,max) + '"></i> ' +
        grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
    }
    return div;
  };

  return [chloro,xlegend];
}


function createMap(crimeSites,
  zipEmploymentRateChloro,EmpRateLegend,
  zipIncomeChloro,InLegend,
  zipADIChloro,ADILegend,
  zipDeathsChloro,DeathsLegend,
  zipOpioidRxChloro,OpioidRxLegend) {

  // Define tile layers
  var darkMap = L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href='https://www.openstreetmap.org/'>OpenStreetMap</a> contributors, <a href='https://creativecommons.org/licenses/by-sa/2.0/'>CC-BY-SA</a>, Imagery © <a href='https://www.mapbox.com/'>Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.dark",
    accessToken: API_KEY
  });
  var streetMap = L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href='https://www.openstreetmap.org/'>OpenStreetMap</a> contributors, <a href='https://creativecommons.org/licenses/by-sa/2.0/'>CC-BY-SA</a>, Imagery © <a href='https://www.mapbox.com/'>Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.streets",
    accessToken: API_KEY
  });
  var lightMap = L.tileLayer("https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}", {
    attribution: "Map data &copy; <a href='https://www.openstreetmap.org/'>OpenStreetMap</a> contributors, <a href='https://creativecommons.org/licenses/by-sa/2.0/'>CC-BY-SA</a>, Imagery © <a href='https://www.mapbox.com/'>Mapbox</a>",
    maxZoom: 18,
    id: "mapbox.light",
    accessToken: API_KEY
  });

  // Define a baseMaps object to hold our base layers
  var baseMaps = {
    "Dark Map": darkMap,
    "Street Map": streetMap,
    "Light Map": lightMap
  };

  // Add crime sites and zip code boundary layers
  var crimeLayer = L.layerGroup(crimeSites);
  var emptyLayer = new L.LayerGroup()

  // Overlays that may be toggled on or off
  var overlayMaps = {
    "LA Crime Sites": crimeLayer,
  };

  var overlayMapsAsBase = {
    "Nothing": emptyLayer,
    "Employment Rate": zipEmploymentRateChloro,
    "Median Income": zipIncomeChloro,
    "Neighborhood Deprivation": zipADIChloro,
    "Deaths per 1,000 Pop": zipDeathsChloro,
    "Opioid Rx per 1,000 Pop": zipOpioidRxChloro
  };

  // Creating map object and set default layers
  var myMap = L.map("map", {
    center: [34.02, -118.3],
    zoom: 10.4,
    layers: [streetMap, crimeLayer, emptyLayer]
  });


  // Pass map layers into layer control
  // Add the layer control to the map
  L.control.layers(baseMaps, overlayMaps, {
    collapsed: false
  }).addTo(myMap);

  var xcontrol = L.control.layers(overlayMapsAsBase,{},{
    collapsed: false
  });
  myMap.addControl(xcontrol);


  // Create legend, this one is for the crime site stuff
  var legendCS = L.control({ position: 'bottomright' });
  legendCS.onAdd = function (myMap) {
    var div = L.DomUtil.create('div', 'info legend'),
      grades = [100, 200, 300, 400, 500, 600, 700],
      labels = ['<strong>Text</strong>'];

    div.innerHTML += '<b>Crime Code</b><br>' 
    // Loop through intervals to generate labels and colored squares for crime site legend
    for (var i = 0; i < grades.length; i++) {
      div.innerHTML +=
        '<i style="background:' + getColorCrimeSites(grades[i] + 1) + '"></i>' +
        grades[i] + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+');
    }
    return div;
  };
  legendCS.addTo(myMap);


  myMap.on('overlayadd', function (eventLayer) {
    if (eventLayer.name === "LA Crime Sites") {
      legendCS.addTo(this);
    }
    else {
      EmpRateLegend.addTo(this);
      this.removeControl(legendCS);
    }
  });

  myMap.on('overlayremove', function (eventLayer) {
    if (eventLayer.name === "LA Crime Sites") {
      this.removeControl(legendCS);
    }
  });

  myMap.on('baselayerchange', function (eventLayer) {
    console.log(eventLayer.name);
    if (eventLayer.name === "Employment Rate"){
      EmpRateLegend.addTo(this);
      this.removeControl(InLegend);
      this.removeControl(ADILegend);
      this.removeControl(DeathsLegend);
      this.removeControl(OpioidRxLegend);
    } else if (eventLayer.name === "Median Income") {
      InLegend.addTo(this);
      this.removeControl(ADILegend);
      this.removeControl(EmpRateLegend);
      this.removeControl(DeathsLegend);
      this.removeControl(OpioidRxLegend);
    } else if (eventLayer.name === "Neighborhood Deprivation") {
      ADILegend.addTo(this);
      this.removeControl(InLegend);
      this.removeControl(EmpRateLegend);
      this.removeControl(DeathsLegend);
      this.removeControl(OpioidRxLegend);
    } else if (eventLayer.name === "Deaths per 1,000 Pop") {
      DeathsLegend.addTo(this);
      this.removeControl(ADILegend);
      this.removeControl(EmpRateLegend);
      this.removeControl(OpioidRxLegend);
      this.removeControl(InLegend);
    } else if (eventLayer.name === "Opioid Rx per 1,000 Pop") {
      OpioidRxLegend.addTo(this);
      this.removeControl(ADILegend);
      this.removeControl(EmpRateLegend);
      this.removeControl(DeathsLegend);
      this.removeControl(InLegend);
    } else if (eventLayer.name === "Nothing") {
      this.removeControl(EmpRateLegend);
      this.removeControl(InLegend);
      this.removeControl(ADILegend);
      this.removeControl(DeathsLegend);
      this.removeControl(OpioidRxLegend);
    }


  });


  //legend.addTo(myMap);

}