require("dotenv").config();
module.exports = {
	packagerConfig: {
		osxSign: {},
		asar: false,
		icon: "assets/icons/icon",
		ignore: [
			"src",
			"tsconfig.json",
			"yarn.lock",
			".env",
			"README.md",
			"LICENSE",
		],
	},
	makers: [
		{
			name: "@electron-forge/maker-zip",
			platforms: ["win32"],
		},
		{
			name: "@electron-forge/maker-dmg",
			config: {
				format: "ULFO",
			},
		},
		{
			name: "@electron-forge/maker-deb",
		},
	],
};
