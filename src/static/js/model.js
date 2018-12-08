
var w = window,
    d = document,
    e = d.documentElement,
    g = d.getElementsByTagName('div#stack')[0],
    x = w.innerWidth || e.clientWidth || g.clientWidth,
    y = w.innerHeight  || e.clientHeight|| g.clientHeight;

var margin = {
  top: 20,
  right: 100,
  bottom: 50,
  left: 200
  };

var width = 900;
var height = 400;

// Create a scalable SVG wrapper, append an SVG group that will hold our chart
var svg = d3.select('div#stack')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMidYMid meet")
   .attr("viewBox", "-40 -10 1350 1350")
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .attr("transform", `translate(${margin.left}, ${margin.top})`); 

// Append an SVG group
// var chartGroup = svg.append("g")
//   .attr("transform", `translate(${margin.left}, ${margin.top})`);

    function wrap(text, width) {
        text.each(function() {
            var text = d3.select(this),
                    words = text.text().split(/\s+/).reverse(),
                    word,
                    line = [],
                    lineNumber = 0,
                    lineHeight = 1.1, // ems
                    y = text.attr("y"),
                    dy = parseFloat(text.attr("dy")),
                    tspan = text.text(null).append("tspan").attr("x", 0).attr("y", y).attr("dy", dy + "em");
            while (word = words.pop()) {
                line.push(word);
                tspan.text(line.join(" "));
                if (tspan.node().getComputedTextLength() > width) {
                    line.pop();
                    tspan.text(line.join(" "));
                    line = [word];
                    tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
                }
            }
        });
    }
    dataset = [
            {label:"Homicide","Accurate":0,"Inaccurate":100},
            {label:"Rape","Accurate":86,"Inaccurate":14},
            {label:"Aggravated Assault","Accurate":96,"Inaccurate":4},
            {label:"Robbery","Accurate":86,"Inaccurate":14},
            {label:"Burglary","Accurate":86,"Inaccurate":14},
            {label:"Burglary/Theft From Vehicle","Accurate":74,"Inaccurate":26},
            {label:"Motor Vehicle Theft","Accurate":88,"Inaccurate":12},
            {label:"Personal/Other Theft","Accurate":89,"Inaccurate":11} ];

    var x = d3.scale.ordinal()
            .rangeRoundBands([0, width], .1,.3);
    var y = d3.scale.linear()
            .rangeRound([height, 0]);
    var colorRange = d3.scale.category20c().range(["#383838", "#ffffff"]);
    var color = d3.scale.ordinal()
            .range(colorRange.range());
    var xAxis = d3.svg.axis()
            .scale(x)
            .orient("bottom");
    var yAxis = d3.svg.axis()
            .scale(y)
            .orient("left")
            .tickFormat(d3.format(".2s"));
    // var svg = d3.select("div#stack").append("svg")
    //         .attr("width", width + margin.left + margin.right)
    //         .attr("height", height + margin.top + margin.bottom)
    //         .append("g")
    //         .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    var divTooltip = d3.select("div#stack").append("div").attr("class", "toolTip");
    color.domain(d3.keys(dataset[0]).filter(function(key) { return key !== "label"; }));
    dataset.forEach(function(d) {
        var y0 = 0;
        d.values = color.domain().map(function(name) { return {name: name, y0: y0, y1: y0 += +d[name]}; });
        d.total = d.values[d.values.length - 1].y1;
    });
    x.domain(dataset.map(function(d) { return d.label; }));
    y.domain([0, d3.max(dataset, function(d) { return d.total; })]);
    svg.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + height + ")")
            .call(xAxis);
    svg.append("g")
            .attr("class", "y axis")
            .call(yAxis)
            .append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 9)
            .attr("dy", ".71em")
            .style("text-anchor", "end")
            .text("Model Accuracy %");
    var bar = svg.selectAll(".label")
            .data(dataset)
            .enter().append("g")
            .attr("class", "g")
            .attr("transform", function(d) { return "translate(" + x(d.label) + ",0)"; });
    svg.selectAll(".x.axis .tick text")
            .call(wrap, x.rangeBand());
            
    var bar_enter = bar.selectAll("rect")
    .data(function(d) { return d.values; })
    .enter();

bar_enter.append("rect")
    .attr("width", x.rangeBand())
    .attr("y", function(d) { return y(d.y1); })
    .attr("height", function(d) { return y(d.y0) - y(d.y1); })
    .style("fill", function(d) { return color(d.name); });

bar_enter.append("text")
    .text(function(d) { return d3.format("")(d.y1-d.y0) + "%"; })
    .attr("y", function(d) { return y(d.y1)+(y(d.y0) - y(d.y1))/2; })
    .attr("x", x.rangeBand()/2.5)
    .style("fill", '#ffffff');
    
    bar
            .on("mousemove", function(d){
                divTooltip.style("left", d3.event.pageX+10+"px");
                divTooltip.style("top", d3.event.pageY-25+"px");
                divTooltip.style("display", "inline-block");
                var elements = document.querySelectorAll(':hover');
                l = elements.length
                l = l-1
                element = elements[l].__data__
                value = element.y1 - element.y0
                divTooltip.html((d.label)+"<br>"+element.name+"<br>"+value+"%");
            });
    bar
            .on("mouseout", function(d){
                divTooltip.style("display", "none");
            });
    // svg.append("g")
    //         .attr("class", "legendLinear")
    //         .attr("transform", "translate(50, 50)");
    // var legend = d3.legend.color()
    //         .shapeWidth(height/4)
    //         .shapePadding(10)
    //         .orient('horizontal')
    //         .scale(color);
    // svg.select(".legendLinear")
    //         .call(legend);

