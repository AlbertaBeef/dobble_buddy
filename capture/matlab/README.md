# Dobble capture script for use with Matlab

Requirements:
- MATLAB
- Webcam Support Package
   - if not already installed, will open the Add-On Explorer to install

Edit script for your setup:
- specify webcam by number (ie. 0, 1, ...) or by name
   - list of cameras can be obtained with "webcamlist" command in MATLAB
```
>> webcamlist

ans =
 
  2×1 cell array

    {'HP HD Camera'  }
    {'HD Pro Webcam C920'}
```

Instructions:
- Script will display this interactive figure
	<div align="center">
    <img width="50%" height="50%" src="img/capture_gui_01.png">
	</div>
		
- In order to capture an image, card must be fully visible.
- The following examples are invalid
	<div align="center">
    <img width="50%" height="50%" src="img/capture_invalid_01.png">
    <img width="50%" height="50%" src="img/capture_invalid_02.png">
	</div>
		
- When valid, the card will have a BLUE RECTANGLE drawn around it, as shown here
	<div align="center">
    <img width="50%" height="50%" src="img/capture_gui_02.png">
	</div>
		
- To capture image, press to the left of the "Dobble - Capturing Card #" title, as indicated by the red arrow & box shown below … 
	<div align="center">
    <img width="50%" height="50%" src="img/capture_gui_03.png">
	</div>
		
- Captured image should look like this …
	<div align="center">
    <img width="100%" height="100%" src="img/capture_final_01.tif">
	</div>
		
