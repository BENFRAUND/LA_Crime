var buckets = ["accurate", "inaccurate", "total"];

var margin = {top: 20, right: 50, bottom: 30, left: 20},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

var x = d3.scale.category10()
    .rangeRoundBands([0, width]);

var y = d3.scale.linear()
    .rangeRound([height, 0]);

var z = d3.scale.category10();

var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")
    .tickFormat(d3.time.format("%b"));

var yAxis = d3.svg.axis()
    .scale(y)
    .orient("left");

var svg = d3.select('#stack').append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// Retrieve data from the API and execute everything below
d3.json("/crime_stats/results", function(crimeAccuracy) {
     return crimeAccuracy.map(function(d) {
         return {x: d.fbi_category, y: d[c]};
     });
    });

//       data.fbi_category = data.FBI_Category;
//       data.accurate = +data.Accurate;
//       data.inaccurate = +data.Inaccurate;
//       data.total = +data.Total;
      
//       console.log(data.fbi_category);

// });

// d3.csv("model_accuracy.csv", type, function(error, crimeAccuracy) {
//   if (error) throw error;

//   var layers = d3.layout.stack()(buckets.map(function(c) {
//     return crimeAccuracy.map(function(d) {
//       return {x: d.fbi_category, y: d[c]};
//     });
//   }));

  x.domain(layers[0].map(function(d) { return d.x; }));
  y.domain([0, d3.max(layers[layers.length - 1], function(d) { return d.y0 + d.y; })]).nice();

  var layer = svg.selectAll(".layer")
      .data(layers)
    .enter().append("g")
      .attr("class", "layer")
      .style("fill", function(d, i) { return z(i); });

  layer.selectAll("rect")
      .data(function(d) { return d; })
    .enter().append("rect")
      .attr("x", function(d) { return x(d.x); })
      .attr("y", function(d) { return y(d.y + d.y0); })
      .attr("height", function(d) { return y(d.y0) - y(d.y + d.y0); })
      .attr("width", x.rangeBand() - 1);

  svg.append("g")
      .attr("class", "axis axis--x")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

  svg.append("g")
      .attr("class", "axis axis--y")
      .attr("transform", "translate(" + width + ",0)")
      .call(yAxis);


function type(d) {
  d.fbi_category = d.fbi_category;
  buckets.forEach(function(c) { d[c] = +d[c]; });
  return d;
}