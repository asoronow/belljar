var ipc = require("electron").ipcRenderer;
var run = document.getElementById("run");
var indir = document.getElementById("indir");
var outdir = document.getElementById("outdir");
var loadbar = document.getElementById("loadbar");
var loadmessage = document.getElementById("loadmessage");
var back = document.getElementById("back");
var whole = document.getElementById("whole");
var half = document.getElementById("half");
var angle = document.getElementById("angle");
var alignmentMethod = "True";
var methods = document.querySelector("#methods");

whole.addEventListener("click", function () {
	methods.textContent = "Whole Slice";
	alignmentMethod = "True";
	console.log(alignmentMethod);
});

half.addEventListener("click", function () {
	methods.textContent = "Hemisphere Only";
	alignmentMethod = "False";
	console.log(alignmentMethod);
});

run.addEventListener("click", function () {
	if (indir && outdir && indir.value && outdir.value) {
		var a;
		if (angle && angle.value && angle.value >= -10 && angle.value <= 10) {
			a = angle.value;
		} else if (!angle || !angle.value) {
			a = 99;
		} else {
			alert("Please enter a valid angle between -10 and 10");
			return;
		}

		run.classList.add("disabled");
		back.classList.remove("btn-warning");
		back.classList.add("btn-danger");
		back.innerHTML = "Cancel";
		run.innerHTML = "<i class='fas fa-spinner fa-spin'></i>";
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
