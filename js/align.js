const { t } = require("tar");

var ipc = require("electron").ipcRenderer;
var run = document.getElementById("run");
var indir = document.getElementById("indir");
var outdir = document.getElementById("outdir");
var loadbar = document.getElementById("loadbar");
var loadmessage = document.getElementById("loadmessage");
var back = document.getElementById("back");
var whole = document.getElementById("whole");
var half = document.getElementById("half");
var spacing = document.getElementById("spacing");
var alignmentMethod = "True";
var methods = document.querySelector("#methods");

whole.addEventListener("click", function () {
	methods.textContent = "Both Hemispheres";
	alignmentMethod = "True";
	console.log(alignmentMethod);
});

half.addEventListener("click", function () {
	methods.textContent = "Single Hemisphere";
	alignmentMethod = "False";
	console.log(alignmentMethod);
});

run.addEventListener("click", function () {
	if (indir && outdir && indir.value && outdir.value) {
		var a = spacing.value;
		// use try catch to check if a is a number
		try {
			a = Number(a);
		} catch (err) {
			console.log(err);
			alert("Spacing must be a integer!");
			return;
		}

		if (a % 1 != 0) {
			a = Math.round(a);
		}

		run.classList.add("disabled");
		back.classList.remove("btn-warning");
		back.classList.add("btn-danger");
		back.innerHTML = "Cancel";
		run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
		loadmessage.innerHTML = "Intializing...";
		ipc.send("runAlign", [indir.value, outdir.value, alignmentMethod, a]);
	}
});

back.addEventListener("click", function (event) {
	if (back.classList.contains("btn-danger")) {
		event.preventDefault();
		ipc.send("killAlign", []);
		back.classList.add("btn-warning");
		back.classList.remove("btn-danger");
		back.innerHTML = "Back";
		run.innerHTML = "Run";
		run.classList.remove("disabled");
		loadmessage.innerHTML = "";
		loadbar.style.width = "0";
	}
});

ipc.on("alignResult", function (event, response) {
	run.innerHTML = "Run";
	run.classList.remove("disabled");
	back.classList.add("btn-warning");
	back.classList.remove("btn-danger");
	back.innerHTML = "Back";
	run.innerHTML = "Run";
	run.classList.remove("disabled");
	loadmessage.innerHTML = "";
	loadbar.style.width = "0";
});

ipc.once("alignError", function (event, response) {
	run.innerHTML = "Run";
	run.classList.remove("disabled");
});

ipc.on("updateLoad", function (event, response) {
	loadbar.style.width = String(response[0]) + "%";
	loadmessage.innerHTML = response[1];
});

indir.addEventListener("click", function () {
	ipc.once("returnPath", function (event, response) {
		if (response[1] == "indir") {
			indir.value = response[0];
		}
	});
	ipc.send("openDialog", "indir");
});

outdir.addEventListener("click", function () {
	ipc.once("returnPath", function (event, response) {
		if (response[1] == "outdir") {
			outdir.value = response[0];
		}
	});
	ipc.send("openDialog", "outdir");
});
