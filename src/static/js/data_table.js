//create Tabulator on DOM element with id "crime-table"
  var table = new Tabulator("#crime-table", {
 	//height:500, // set height of table (in CSS or here), this enables the Virtual DOM and improves render speed dramatically (can be any valid css height value)
	//height:"100%",
	pagination:"local", //enable local pagination.
    paginationSize:15, // rows per page (default = 10)
	layout:"fitColumns", //fit columns to width of table (optional)
	responsiveLayout:"hide",
 	columns:[ //Define Table Columns
		{title:"DR#", field:"dr_no", align:"left", width:100},
		{title:"Area", field:"area_name", width:80, align:"left", headerFilter: true},
		{title:"Location", field:"location", align:"left", headerFilter: true, width:120},
	 	{title:"Cross Steet", field:"cross_street", align:"left", width:150},
	 	{title:"Code", field:"crm_cd", align:"left", sorter:"number", width:70},
	 	{title:"Crime Description", field:"crm_cd_desc", align:"left", headerFilter:true},
	 	{title:"Date Occurred", field:"date_occ", align:"left", headerFilter:true, width:130},
		{title:"Hour", field:"hour_occ", align:"leftr", sorter:"number", width:70},
		{title:"Premises", field:"premis_desc", align:"left"},
		{title:"Status", field:"status_desc", align:"left", width:100},
		{title:"Weapon", field:"weapon_desc", align:"left", headerFilter:true}
		 
 	],
});


$("#tabulator-controls  button[name=download]").on("click", function(){
   	table.download("csv", "la_crimes.csv");
});


d3.json("/crime_sites", function(data) {
    //load sample data into the table
   table.setData(data);
  
});
