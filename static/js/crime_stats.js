var w = window,
    d = document,
    e = d.documentElement,
    g = d.getElementsByTagName('div#scatter')[0],
    x = w.innerWidth || e.clientWidth || g.clientWidth,
    y = w.innerHeight  || e.clientHeight|| g.clientHeight;

var margin = {
  top: 0,
  right: 0,
  bottom: 0,
  left: 330
  };

var width = 700;
var height = 420;

// Create a scalable SVG wrapper, append an SVG group that will hold our chart
var svg = d3.select('div#scatter')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMidYMid meet")
   .attr("viewBox", "0 0 1350 1350")
   //class to make it responsive
   .classed("svg-content-responsive", true); 

// Append an SVG group
var chartGroup = svg.append("g")
  .attr("transform", `translate(${margin.left}, ${margin.top})`);

// Initial Params
var chosenXAxis = "crime_sites";
var chosenYAxis = "housing_prices";
var correlation = -0.53  // initial correlation value needs to change if initial x and y variables change

// Title case function
function titleCase(str) {
  str = str.toLowerCase().split('_');
  for (var i = 0; i < str.length; i++) {
    str[i] = str[i].charAt(0).toUpperCase() + str[i].slice(1); 
  }
  return str.join(' ');
};

Title = titleCase(`${chosenXAxis}`) + " and " + titleCase(`${chosenYAxis}`);
subTitle = `(r = ${correlation})`;
d3.select("#Title").html(Title);
d3.select("#subTitle").html(subTitle);
//------------------------------------------------------------------

// function used for updating x-scale var upon click on axis label
function xScale(statesData, chosenXAxis) {
  // create scales
  var xLinearScale = d3.scaleLinear()
    .domain([d3.min(statesData, d => d[chosenXAxis]) * 0.8,
      d3.max(statesData, d => d[chosenXAxis]) * 1.2
    ])
    .range([0, width]);

  return xLinearScale;

}

// function used for updating y-scale var upon click on axis label
function yScale(statesData, chosenYAxis) {
  // create scales
  var yLinearScale = d3.scaleLinear()
    .domain([d3.min(statesData, d => d[chosenYAxis]) * 0.8,
      d3.max(statesData, d => d[chosenYAxis]) * 1.2
    ])
    .range([height, 0]);

  return yLinearScale;
}
//------------------------------------------------------------------

// function used for updating xAxis var upon click on axis label
function renderXAxes(newXScale, xAxis) {
  var bottomAxis = d3.axisBottom(newXScale);

  xAxis.transition()
    .duration(1000)
    .call(bottomAxis);

  return xAxis;
}

// function used for updating yAxis var upon click on axis label
function renderYAxes(newYScale, yAxis) {
  var leftAxis = d3.axisLeft(newYScale);

  yAxis.transition()
    .duration(1000)
    .call(leftAxis);

  return yAxis;
}
//-----------------------------------------------------------------

// function used for updating circles group with a transition to
// new circles
function renderCircles(circlesGroup, newXScale, chosenXAxis, newYScale, chosenYAxis) {

  circlesGroup.transition()
    .duration(1000)
    .attr("cx", d => newXScale(d[chosenXAxis]))
    .attr("cy", d => newYScale(d[chosenYAxis]));
    
  return circlesGroup;
}


// function used for moving State abbreviation labels along with circles
function renderCircleText(circlesText, newXScale, chosenXAxis, newYScale, chosenYAxis) {

  circlesText.transition()
    .duration(1000)
    .attr("x", d => newXScale(d[chosenXAxis]))
    .attr("y", d => newYScale(d[chosenYAxis]));

  return circlesText;

}

//----------------------------------------------------------------

// function used for updating circles group with new tooltip
function updateToolTip(chosenXAxis, chosenYAxis, circlesText) {

  switch (chosenXAxis) {
    case "superfund_sites":
      var xlabel = "Superfund Sites";
      var xformat = d3.format("");
      var xsuffix = "";
      break;
    case "smoking":
      var xlabel = "Smoking";
      var xformat = d3.format("");
      var xsuffix = "%";
      break;
    case "obesity":
      var xlabel = "Obesity";
      var xformat = d3.format("");
      var xsuffix = "%";
  }

  switch (chosenYAxis) {
    case "cancer_mortality":
      var ylabel = "Cancer Mortality";
      var yformat = d3.format(",");
      var ysuffix = "";
      break;
    case "income":
      var ylabel = "Income";
      var yformat = d3.format("$,");
      var ysuffix = "";
      break;
    case "population_density":
      var ylabel = "Population Density";
      var yformat = d3.format("");
      var ysuffix = "";
  }
  
  var toolTip = d3.tip()
    .attr("class", "tooltip")
    .offset([80, -60])
    .html(function(d) {
      return (`${d.state.bold()}<br>${xlabel}: ${xformat(d[chosenXAxis])}${xsuffix}<br>${ylabel}: ${yformat(d[chosenYAxis])}${ysuffix}`);
    });

  circlesText.call(toolTip);

  circlesText.on("mouseover", function(data) {
    toolTip.show(data)
  })
    // onmouseout event
    .on("mouseout", function(data, index) {
      toolTip.hide(data);
  });
   
  return circlesText;

}

//----------------------------------------------------------------

// Retrieve data from the API and execute everything below
d3.json("/crime_stats/get_data").then(function(statesData) {
    statesData.forEach(function(data) {
      data.state = data.state;
      data.abbr = data.state_abbr;
      data.poverty = +data.pct_below_poverty;
      data.age = +data.median_age;
      data.income = +data.median_household_income;
      data.veterans = +data.pct_veterans;
      data.population = +data.population;
      data.housing_density = +data.housing_density;
      data.population_density = +data.population_density;
      data.cancer_mortality = +data.cancer_death_rate;
      data.cancer_morbidity = +data.cancer_prevalence_rate;
      data.cancer_incidence = +data.cancer_incidence_rate;
      data.superfund_sites = +data.sf_site_count;
      data.superfund_score = +data.avg_hrsscore;
      data.smoking = +data.smokes;
      data.obesity = +data.obesity;

      console.log(data.smoking);

});

  var correlation = ss.sampleCorrelation(statesData.map(d => d[chosenXAxis]), statesData.map(d => d[chosenYAxis])).toFixed(2);

// xLinearScale function above csv import
  var xLinearScale = xScale(statesData, chosenXAxis);

  // Create y scale function
  var yLinearScale = yScale(statesData, chosenYAxis);

  // Create initial axis functions
  var bottomAxis = d3.axisBottom(xLinearScale);
  var leftAxis = d3.axisLeft(yLinearScale);

  // append x axis
  var xAxis = chartGroup.append("g")
    .classed("x-axis", true)
    .attr("transform", `translate(0, ${height})`)
    .call(bottomAxis);

  // append y axis
  var yAxis = chartGroup.append("g")
    .classed("y-axis", true)
    //.attr("transform", `translate(0, 0)`)  
    .call(leftAxis);

  // append initial circles
  var circlesGroup = chartGroup.selectAll("circle")
    .data(statesData)
    .enter()
    .append("circle")
    .attr("cx", d => xLinearScale(d[chosenXAxis]))
    .attr("cy", d => yLinearScale(d[chosenYAxis]))
    .attr("r", 15)
    .attr("fill", "purple")
    .attr("opacity", ".6");

  // append initial circle text
  var circlesText = chartGroup.selectAll("circlesGroup")
    .data(statesData)
    .enter()
    .append("text")
    .text(function(data) {return data.abbr})
    .attr("x", d => xLinearScale(d[chosenXAxis]))
    .attr("y", d => yLinearScale(d[chosenYAxis]))
    .attr("dy", ".35em")
    // .attr("dz", "1")
    .attr("font-size", "10px")
    .attr("text-anchor", "middle")
    .attr("class", "stateText");
    
  // Create group for  3 x-axis labels
  var xLabelsGroup = chartGroup.append("g")
    .attr("transform", `translate(${width / 2}, ${height + 20})`);

  var superfundCntLabel = xLabelsGroup.append("text")
    .attr("x", 0)
    .attr("y", 20)
    .attr("value", "superfund_sites") // value to grab for event listener
    .classed("active", true)
    .text("Number of Superfund Sites");

  var popDensityLabel = xLabelsGroup.append("text")
    .attr("x", 0)
    .attr("y", 40)
    .attr("value", "smoking") // value to grab for event listener
    .classed("inactive", true)
    .text("Smoking (%)");

  var incomeLabel = xLabelsGroup.append("text")
    .attr("x", 0)
    .attr("y", 60)
    .attr("value", "obesity") // value to grab for event listener
    .classed("inactive", true)
    .text("Obesity (%)");

    // Create group for 3 y-axis labels

  var yLabelsGroup = chartGroup.append("g")
    .attr("transform", `translate(${0}, ${(height / 2) - 20})`);

  var cancerMortalityLabel = yLabelsGroup.append("text")
    .attr("transform", "rotate(-90)")  
    .attr("y", -50)
    .attr("x", 0)
    .attr("value", "cancer_mortality") // value to grab for event listener
    .classed("active", true)
    .text("Cancer Mortality (per 100,000)");
  
  var cancerPrevalenceLabel = yLabelsGroup.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", -70)
    .attr("x", 0)
    .attr("value", "population_density") // value to grab for event listener
    .classed("inactive", true)
    .text("Population Density (per sq mile)");

  var pctVeteransLabel = yLabelsGroup.append("text")
    .attr("transform", "rotate(-90)")  
    .attr("y", -90)
    .attr("x", 0)
    .attr("value", "income") // value to grab for event listener
    .classed("inactive", true)
    .text("Household Income (Median)");

  // updateToolTip function above csv import
  var circlesGroup = updateToolTip(chosenXAxis, chosenYAxis, circlesGroup);

  // x axis labels event listener
  xLabelsGroup.selectAll("text")
    .on("click", function() {
      // get value of selection
      var value = d3.select(this).attr("value");
      if (value !== chosenXAxis) {

        // replaces chosenXAxis with value
        chosenXAxis = value;
      
        // replaces correlation value
        correlation = ss.sampleCorrelation(statesData.map(d => d[chosenXAxis]), statesData.map(d => d[chosenYAxis])).toFixed(2);

        // functions here found above csv import
        // updates x scale for new data
        xLinearScale = xScale(statesData, chosenXAxis);

        // updates x axis with transition
        xAxis = renderXAxes(xLinearScale, xAxis);

        // updates circles with new x values
        circlesGroup = renderCircles(circlesGroup, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

        // updates circle text with new x values
        circlesText = renderCircleText(circlesText, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);      //bookmark//

        // updates tooltips with new info
        circlesGroup = updateToolTip(chosenXAxis, chosenYAxis, circlesGroup);

        Title = titleCase(`${chosenXAxis}`) + " and " + titleCase(`${chosenYAxis}`);
        subTitle = `(r = ${correlation})`;
        d3.select("#Title").html(Title);
        d3.select("#subTitle").html(subTitle);
        
        // changes classes to change bold text
        switch (chosenXAxis) {
          case "superfund_sites":
            superfundCntLabel
              .classed("active", true)
              .classed("inactive", false);
            popDensityLabel
              .classed("active", false)
              .classed("inactive", true);
            incomeLabel
              .classed("active", false)
              .classed("inactive", true);
              break;
          case "smoking":
            superfundCntLabel
              .classed("active", false)
              .classed("inactive", true);
            popDensityLabel
              .classed("active", true)
              .classed("inactive", false);
            incomeLabel
              .classed("active", false)
              .classed("inactive", true);
              break;
          case "obesity":
            superfundCntLabel
              .classed("active", false)
              .classed("inactive", true);
            popDensityLabel
              .classed("active", false)
              .classed("inactive", true);
            incomeLabel
              .classed("active", true)
              .classed("inactive", false);
        }
      }
    });

  yLabelsGroup.selectAll("text")
    .on("click", function() {
    // get value of selection
      var value = d3.select(this).attr("value");
      if (value !== chosenYAxis) {

        // replaces chosenXAxis with value
        chosenYAxis = value;

        // replaces correlation value
        correlation = ss.sampleCorrelation(statesData.map(d => d[chosenXAxis]), statesData.map(d => d[chosenYAxis])).toFixed(2);

        // functions here found above csv import
        // updates x scale for new data
        yLinearScale = yScale(statesData, chosenYAxis);

        // updates y axis with transition
        yAxis = renderYAxes(yLinearScale, yAxis);

        // updates circles with new y values
        circlesGroup = renderCircles(circlesGroup, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

        // updates circle text with new y values
        circlesText = renderCircleText(circlesText, xLinearScale, chosenXAxis, yLinearScale, chosenYAxis);

        // updates tooltips with new info
        circlesGroup = updateToolTip(chosenXAxis, chosenYAxis, circlesGroup);

        Title = titleCase(`${chosenXAxis}`) + " and " + titleCase(`${chosenYAxis}`);
        subTitle = `(r = ${correlation})`;
        d3.select("#Title").html(Title);
        d3.select("#subTitle").html(subTitle);

        // changes classes to change bold text
        switch (chosenYAxis) {
          case "cancer_mortality":
            cancerMortalityLabel
              .classed("active", true)
              .classed("inactive", false);
            cancerPrevalenceLabel
              .classed("active", false)
              .classed("inactive", true);
            pctVeteransLabel
              .classed("active", false)
              .classed("inactive", true);
              break;
          case "population_density":
            cancerMortalityLabel
              .classed("active", false)
              .classed("inactive", true);
            cancerPrevalenceLabel
              .classed("active", true)
              .classed("inactive", false);
            pctVeteransLabel
              .classed("active", false)
              .classed("inactive", true);
              break;
          case "income":
            cancerMortalityLabel
              .classed("active", false)
              .classed("inactive", true);
            cancerPrevalenceLabel
              .classed("active", false)
              .classed("inactive", true);
            pctVeteransLabel
              .classed("active", true)
              .classed("inactive", false);
        }
      }
    });
});
