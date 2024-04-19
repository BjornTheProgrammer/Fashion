import Jimp from 'jimp';
import fs from 'node:fs';
import path from 'path';

// This function takes an image location and a label and makes a csv file with the image data and the label
async function csvImage(imageLocation: string) {
	const image = await Jimp.read(imageLocation);

	let row = '';

	// Loop through every pixel in the image and add it to the row
	image.scan(0, 0, image.bitmap.width, image.bitmap.height, function (x, y, idx) {
		var pixelValue = this.bitmap.data[idx + 1];
		row += pixelValue + ',';
	});

	const label = imageLocation.split('/').at(-1)!.split('-')[0];

	row += `${label}\n`; // Add the label to the end of the row

	const csvLocation = path.resolve(imageLocation, '../..', `csvs/${label}.csv`);
	fs.appendFileSync(csvLocation, row, 'utf-8'); // Write the row to the file
}

async function processDirectory() {
	if (fs.existsSync('./data/csvs')) fs.rmSync('./data/csvs', { recursive: true, force: true });
	fs.mkdirSync('./data/csvs');

	const files = await fs.promises.readdir('./data/images/'); // Get all the files in the directory

	for (const file of files) { // Loop through all the files
		await csvImage(path.resolve('./data/images', file)); // Process the image
	}
}

processDirectory();

