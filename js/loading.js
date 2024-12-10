var ipc = require("electron").ipcRenderer;
var guide = document.getElementById("guide");
let lastUpdateTimestamp = 0;

ipc.on("updateStatus", function (event, response) {
	if (!response.timestamp || response.timestamp > lastUpdateTimestamp) {
		if (response.timestamp) {
			lastUpdateTimestamp = response.timestamp;
		}
		var status = document.getElementById("status");
		status.innerHTML = response.message || response;
	}
});

// get version from ipc
ipc.send("getVersion");
ipc.on("version", function (event, response) {
	var version = document.getElementById("version");
	version.innerHTML = response;
});

guide.addEventListener("click", function () {
	ipc.send("openGuide");
});
