// Global state
let eyeData = { x: [], y: [], time: [] };
let rawFileContent = null;
let smoothedEyeData = { x: [], y: [] };
let velocityData = [];
let smoothingSettings = { enabled: false, windowSize: 5 };
let saccadeMarkingMode = false;
let markedSaccades = []; // Stores {startIndex, endIndex, shapeId}
let labels = [];


function applyMovingAverage(dataArray, windowSize) {
    if (windowSize % 2 === 0) {
        console.warn(`Window size ${windowSize} is even, incrementing to ${windowSize + 1}`);
        windowSize++;
    }
    if (windowSize < 3) windowSize = 3; // Minimum practical window size

    const smoothedArray = [];
    const halfWindow = Math.floor(windowSize / 2);

    for (let i = 0; i < dataArray.length; i++) {
        let sum = 0;
        let count = 0;
        for (let j = -halfWindow; j <= halfWindow; j++) {
            const index = i + j;
            if (index >= 0 && index < dataArray.length) {
                sum += dataArray[index];
                count++;
            }
        }
        smoothedArray.push(count > 0 ? sum / count : dataArray[i]); // Push original if count is 0 (should not happen with proper window)
    }
    return smoothedArray;
}

function calculateVelocity(xPos, yPos) {
    const velArray = [];
    if (xPos.length !== yPos.length || xPos.length < 2) {
        return velArray; // Not enough data or mismatched lengths
    }
    for (let i = 1; i < xPos.length; i++) {
        const diffX = xPos[i] - xPos[i-1];
        const diffY = yPos[i] - yPos[i-1];
        velArray.push(Math.sqrt(diffX**2 + diffY**2));
    }
    return velArray;
}


function initPlots() {
    const initialLayout = (title, yAxisTitle) => ({
        title: title,
        xaxis: { title: 'Time (samples)' },
        yaxis: { title: yAxisTitle },
        datarevision: 0 
    });

    Plotly.newPlot('position-plot', [{ x: [], y: [], mode: 'lines', name: 'X Position Raw'}, { x: [], y: [], mode: 'lines', name: 'Y Position Raw'}], initialLayout('Eye Position', 'Position'));
    Plotly.newPlot('velocity-plot', [{ x: [], y: [], mode: 'lines', name: 'Velocity'}], initialLayout('Eye Velocity', 'Velocity'));
    
    // Initialize smoothing settings from controls
    const smoothingCheckbox = document.getElementById('enable-smoothing');
    const windowInput = document.getElementById('smoothing-window');
    if (smoothingCheckbox) smoothingSettings.enabled = smoothingCheckbox.checked;
    if (windowInput) smoothingSettings.windowSize = parseInt(windowInput.value, 10);

    processAndPlotData(); // Call to set initial plot state based on defaults
}


function processAndPlotData() {
    if (!eyeData.x || eyeData.x.length === 0) {
        // Clear plots if no base data
        const initialLayout = (title, yAxisTitle) => ({
            title: title,
            xaxis: { title: 'Time (samples)' },
            yaxis: { title: yAxisTitle },
            shapes: [], // Ensure shapes are cleared
            datarevision: (Plotly.d3.select('#position-plot').layout.datarevision || 0) + 1
        });
        Plotly.react('position-plot', [{ x: [], y: [], mode: 'lines', name: 'X Position Raw'}, { x: [], y: [], mode: 'lines', name: 'Y Position Raw'}], initialLayout('Eye Position (load data)', 'Position'));
        Plotly.react('velocity-plot', [{ x: [], y: [], mode: 'lines', name: 'Velocity'}], initialLayout('Eye Velocity (load data)', 'Velocity'));
        
        smoothedEyeData = { x: [], y: [] };
        velocityData = [];
        labels = [];
        markedSaccades = [];
        // updateSaccadeVisualizations(); // Ensure shapes are visually cleared
        return;
    }

    // Get current smoothing settings from controls
    const smoothingCheckbox = document.getElementById('enable-smoothing');
    const windowInput = document.getElementById('smoothing-window');
    smoothingSettings.enabled = smoothingCheckbox ? smoothingCheckbox.checked : false;
    smoothingSettings.windowSize = windowInput ? parseInt(windowInput.value, 10) : 5;
    
    if (smoothingSettings.windowSize % 2 === 0) { // Ensure odd window size for UI input
        smoothingSettings.windowSize++;
        if(windowInput) windowInput.value = smoothingSettings.windowSize; // Update UI
    }
    if (smoothingSettings.windowSize < 3) {
        smoothingSettings.windowSize = 3;
        if(windowInput) windowInput.value = smoothingSettings.windowSize; // Update UI
    }


    let xForVelocity = eyeData.x;
    let yForVelocity = eyeData.y;

    if (smoothingSettings.enabled) {
        smoothedEyeData.x = applyMovingAverage(eyeData.x, smoothingSettings.windowSize);
        smoothedEyeData.y = applyMovingAverage(eyeData.y, smoothingSettings.windowSize);
        xForVelocity = smoothedEyeData.x;
        yForVelocity = smoothedEyeData.y;
    } else {
        smoothedEyeData = { x: [], y: [] }; 
    }

    velocityData = calculateVelocity(xForVelocity, yForVelocity);

    updatePositionPlot(); // This will also call updateSaccadeVisualizations
    updateVelocityPlot();
}


function parseEyeData(fileContent) {
    rawFileContent = fileContent; 
    const lines = fileContent.split(/\r\n|\n/);

    const newX = [];
    const newY = [];
    const newTime = [];
    let parseErrorCount = 0;
    let successfullyParsedLines = 0;

    lines.forEach((line, index) => {
        line = line.trim();
        if (line === '' || line.startsWith('#')) {
            return; 
        }
        const parts = line.split(/,|\s+|\t+/).filter(p => p !== ""); 

        if (parts.length >= 2) {
            const xVal = parseFloat(parts[0]);
            const yVal = parseFloat(parts[1]);

            if (!isNaN(xVal) && !isNaN(yVal)) {
                newX.push(xVal);
                newY.push(yVal);
                newTime.push(successfullyParsedLines); 
                successfullyParsedLines++;
            } else {
                console.warn(`Parsing error on line ${index + 1}: Non-numeric data "${parts[0]}, ${parts[1]}"`);
                parseErrorCount++;
            }
        } else {
            console.warn(`Parsing error on line ${index + 1}: Not enough columns (found ${parts.length}, expected at least 2)`);
            parseErrorCount++;
        }
    });

    if (successfullyParsedLines > 0) {
        eyeData.x = newX;
        eyeData.y = newY;
        eyeData.time = newTime;

        // Initialize labels and clear previous saccades
        labels = new Array(eyeData.time.length).fill(0);
        markedSaccades = [];
        // updateSaccadeVisualizations will be called by processAndPlotData via updatePositionPlot
        
        processAndPlotData(); 

        if(parseErrorCount > 0) {
            alert(`Data loaded with ${parseErrorCount} parsing errors. Check console for details.`);
        } else {
            alert("Eye data loaded successfully!");
        }
    } else {
        alert("Failed to parse any valid eye data from the file. Please check file format.");
        eyeData = { x: [], y: [], time: [] }; 
        rawFileContent = null;
        processAndPlotData(); // Will clear plots
    }
}

function updatePositionPlot() {
    const traces = [
        {
            x: eyeData.time,
            y: eyeData.x,
            mode: 'lines',
            name: 'X Position (Raw)',
            line: { color: 'blue', dash: 'solid'}
        },
        {
            x: eyeData.time,
            y: eyeData.y,
            mode: 'lines',
            name: 'Y Position (Raw)',
            line: { color: 'red', dash: 'solid'}
        }
    ];

    if (smoothingSettings.enabled && smoothedEyeData.x.length > 0) {
        traces.push({
            x: eyeData.time, // Assuming smoothed data has same time base
            y: smoothedEyeData.x,
            mode: 'lines',
            name: `X Smoothed (w=${smoothingSettings.windowSize})`,
            line: { color: 'lightblue', dash: 'dash' }
        });
        traces.push({
            x: eyeData.time,
            y: smoothedEyeData.y,
            mode: 'lines',
            name: `Y Smoothed (w=${smoothingSettings.windowSize})`,
            line: { color: 'lightcoral', dash: 'dash' }
        });
    }

    const layout = {
        title: 'Eye Position',
        xaxis: { title: 'Time (samples)' },
        yaxis: { title: 'Position' },
        shapes: markedSaccades.map(s => s.shape), // Add shapes to layout
        datarevision: (Plotly.d3.select('#position-plot').layout.datarevision || 0) + 1
    };
    Plotly.react('position-plot', traces, layout);
    // updateSaccadeVisualizations(); // Call this to ensure shapes are correctly drawn/updated
}


function updateSaccadeVisualizations() {
    const currentLayout = document.getElementById('position-plot').layout;
    const newShapes = markedSaccades.map(saccade => ({
        type: 'rect',
        xref: 'x',
        yref: 'paper', // Spans the full y-height of the plot
        x0: eyeData.time[saccade.startIndex],
        x1: eyeData.time[saccade.endIndex],
        y0: 0,
        y1: 1,
        fillcolor: 'rgba(255, 223, 0, 0.3)', // Light yellow
        line: { width: 0 },
        layer: 'below', // Draw below data traces
        name: saccade.shapeId // Optional: for direct manipulation if needed
    }));

    Plotly.relayout('position-plot', { 
        shapes: newShapes,
        datarevision: (currentLayout.datarevision || 0) + 1 
    });
}


function updateVelocityPlot() {
    if (!eyeData.time || eyeData.time.length === 0) { // Check if eyeData.time is valid
        Plotly.react('velocity-plot', [{x:[], y:[]}], {title: 'Eye Velocity (load data)', xaxis: {title: 'Time (samples)'}, yaxis: {title: 'Velocity'}, datarevision: (Plotly.d3.select('#velocity-plot').layout.datarevision || 0) + 1});
        return;
    }
    const timeForVelocity = eyeData.time.slice(1); 
    const traces = [
        {
            x: timeForVelocity,
            y: velocityData,
            mode: 'lines',
            name: 'Velocity'
        }
    ];
    const layout = {
        title: 'Eye Velocity',
        xaxis: { title: 'Time (samples)' },
        yaxis: { title: 'Velocity' },
        datarevision: (Plotly.d3.select('#velocity-plot').layout.datarevision || 0) + 1
    };
    Plotly.react('velocity-plot', traces, layout);
}


// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    initPlots(); 

    const fileInput = document.getElementById('file-input');
    const smoothingCheckbox = document.getElementById('enable-smoothing');
    const windowInput = document.getElementById('smoothing-window');
    const markSaccadeModeButton = document.getElementById('mark-saccade-mode');
    const clearLastSaccadeButton = document.getElementById('clear-last-saccade');
    const clearAllSaccadesButton = document.getElementById('clear-all-saccades');
    const saveLabelsButton = document.getElementById('save-labels-button');
    const positionPlotDiv = document.getElementById('position-plot');


    if (fileInput) {
        fileInput.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (!file) {
                return; // No file selected
            }

            const reader = new FileReader();

            reader.onload = (e) => {
                try {
                    parseEyeData(e.target.result);
                } catch (error) {
                    alert(`An unexpected error occurred during parsing: ${error.message}`);
                    console.error(error);
                     // Reset data and plots in case of critical error
                    eyeData = { x: [], y: [], time: [] };
                    rawFileContent = null;
                    processAndPlotData(); // Will clear plots
                }
            };

            reader.onerror = () => {
                alert(`Error reading file: ${reader.error.message || reader.error}`);
                // Clear file input to allow re-selection of the same file if needed
                fileInput.value = ''; 
            };

            reader.readAsText(file);
        });
    } else {
        console.error("File input element not found!");
    }

    if (smoothingCheckbox) {
        smoothingCheckbox.addEventListener('change', () => {
            smoothingSettings.enabled = smoothingCheckbox.checked;
            processAndPlotData();
        });
    } else {
        console.error("Smoothing checkbox not found!");
    }

    if (windowInput) {
        windowInput.addEventListener('change', () => {
            let newSize = parseInt(windowInput.value, 10);
            if (newSize % 2 === 0) { // Ensure odd
                newSize++;
                windowInput.value = newSize;
            }
            if (newSize < 3) { // Ensure min
                newSize = 3;
                windowInput.value = newSize;
            }
            smoothingSettings.windowSize = newSize;
            processAndPlotData();
        });
    } else {
        console.error("Smoothing window input not found!");
    }

    if (markSaccadeModeButton) {
        markSaccadeModeButton.addEventListener('click', () => {
            saccadeMarkingMode = !saccadeMarkingMode;
            if (saccadeMarkingMode) {
                markSaccadeModeButton.textContent = "Exit Saccade Marking Mode";
                markSaccadeModeButton.style.backgroundColor = "lightgreen"; // Visual feedback
                Plotly.relayout('position-plot', { 'dragmode': 'select' });
            } else {
                markSaccadeModeButton.textContent = "Enter Saccade Marking Mode";
                markSaccadeModeButton.style.backgroundColor = ""; // Revert style
                Plotly.relayout('position-plot', { 'dragmode': 'zoom' });
            }
        });
    } else {
        console.error("Mark Saccade Mode button not found!");
    }

    if (positionPlotDiv) {
        positionPlotDiv.on('plotly_selected', (eventData) => {
            if (!saccadeMarkingMode || !eventData || !eventData.range) {
                // If not in marking mode or selection is cleared, do nothing or clear selection highlights
                if(!saccadeMarkingMode && eventData === undefined){ // Selection was cleared by double click
                     Plotly.relayout('position-plot', {'selections': []}); // Clears the visual selection box
                }
                return;
            }

            const xRange = eventData.range.x;
            const timeMin = xRange[0];
            const timeMax = xRange[1];

            // Convert time values to indices (simple search, could be optimized)
            let startIndex = eyeData.time.findIndex(t => t >= timeMin);
            let endIndex = eyeData.time.findLastIndex(t => t <= timeMax);

            if (startIndex === -1 || endIndex === -1 || startIndex > endIndex) {
                console.warn("Invalid selection range or no data points in range.");
                Plotly.relayout('position-plot', {'selections': []}); // Clears the visual selection box
                return;
            }
            
            // Ensure start is before end
            if (startIndex > endIndex) [startIndex, endIndex] = [endIndex, startIndex];


            const shapeId = `saccade_${new Date().getTime()}`; // Unique ID for the shape
            markedSaccades.push({ startIndex, endIndex, shapeId });
            
            for (let i = startIndex; i <= endIndex; i++) {
                labels[i] = 1;
            }
            
            updateSaccadeVisualizations(); // Redraw all shapes including the new one

            // Clear selection highlight after processing
            Plotly.relayout('position-plot', {'selections': []}); 

            // Optional: Exit marking mode automatically
            // markSaccadeModeButton.click(); 
        });
    } else {
        console.error("Position plot div not found for plotly_selected event!");
    }


    if (clearLastSaccadeButton) {
        clearLastSaccadeButton.addEventListener('click', () => {
            if (markedSaccades.length > 0) {
                const lastSaccade = markedSaccades.pop();
                for (let i = lastSaccade.startIndex; i <= lastSaccade.endIndex; i++) {
                    if (i < labels.length) labels[i] = 0;
                }
                updateSaccadeVisualizations();
            } else {
                alert("No saccades to clear.");
            }
        });
    } else {
        console.error("Clear Last Saccade button not found!");
    }

    if (clearAllSaccadesButton) {
        clearAllSaccadesButton.addEventListener('click', () => {
            if (markedSaccades.length === 0 && !labels.some(l => l === 1) ) {
                alert("No saccades to clear.");
                return;
            }
            markedSaccades = [];
            labels.fill(0);
            updateSaccadeVisualizations();
        });
    } else {
        console.error("Clear All Saccades button not found!");
    }

    if (saveLabelsButton) {
        saveLabelsButton.addEventListener('click', saveLabelsToFile);
    } else {
        console.error("Save Labels button not found!");
    }

});

function saveLabelsToFile() {
    if (!labels || labels.length === 0) {
        alert("No label data to save.");
        return;
    }

    // Prepare content: each label on a new line
    const labelString = labels.join('\n');

    // Create Blob and URL
    const blob = new Blob([labelString], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    // Create temporary anchor element and trigger download
    const anchorElement = document.createElement('a');
    anchorElement.href = url;
    anchorElement.download = "saccade_labels.csv"; // Default filename
    document.body.appendChild(anchorElement); // Append to body to make it clickable
    anchorElement.click();

    // Cleanup
    document.body.removeChild(anchorElement);
    URL.revokeObjectURL(url);

    alert("Labels file prepared for download.");
}
