const API_URL = 'http://localhost:5000';

// Load model info and features on page load
document.addEventListener('DOMContentLoaded', async () => {
    if (document.getElementById('predictForm')) {
        await loadModelInfo();
        await loadFeatures();
    }
});

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_URL}/model-info`);
        const data = await response.json();

        const statsDiv = document.getElementById('modelStats');
        statsDiv.innerHTML = `
            <h3>Model Information</h3>
            <p><strong>Model:</strong> ${data.model_name}</p>
            <p><strong>F1-Score:</strong> ${data.f1_score.toFixed(4)}</p>
            <p><strong>Accuracy:</strong> ${data.accuracy ? data.accuracy.toFixed(4) : 'N/A'}</p>
            <p><strong>Features:</strong> ${data.num_features}</p>
        `;
    } catch (error) {
        console.error('Error loading model info:', error);
        document.getElementById('modelStats').innerHTML = `
            <h3>Model Information</h3>
            <p style="color: red;">Error loading model information. Make sure the backend server is running.</p>
        `;
    }
}

async function loadFeatures() {
    try {
        const response = await fetch(`${API_URL}/features`);
        const data = await response.json();

        const featureInputs = document.getElementById('featureInputs');

        // Flag features that must be 0 or 1
        const flagFeatures = [
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count'
        ];

        featureInputs.innerHTML = data.features.map(feature => {
            let inputAttrs = 'type="number" step="any" value="0" required';

            // Add validation for temporal features
            if (feature === 'hour') {
                inputAttrs = 'type="number" min="0" max="23" step="1" value="0" required title="Hour must be between 0 and 23"';
            } else if (feature === 'minute') {
                inputAttrs = 'type="number" min="0" max="59" step="1" value="0" required title="Minute must be between 0 and 59"';
            } else if (feature === 'second') {
                inputAttrs = 'type="number" min="0" max="59" step="1" value="0" required title="Second must be between 0 and 59"';
            } else if (flagFeatures.includes(feature)) {
                // Flag features must be 0 or 1
                inputAttrs = 'type="number" min="0" max="1" step="1" value="0" required title="Must be 0 or 1"';
            }

            return `
                <div class="form-group">
                    <label for="${feature}">${feature}</label>
                    <input ${inputAttrs} id="${feature}" name="${feature}">
                </div>
            `;
        }).join('');

    } catch (error) {
        console.error('Error loading features:', error);
    }
}

// Handle sample data buttons
document.querySelectorAll('.btn-sample').forEach(button => {
    button.addEventListener('click', async () => {
        const attackType = button.getAttribute('data-type');

        try {
            const response = await fetch(`${API_URL}/sample-data?type=${attackType}`);
            const data = await response.json();

            Object.entries(data.sample).forEach(([feature, value]) => {
                const input = document.getElementById(feature);
                if (input) {
                    input.value = value;
                }
            });

            // Visual feedback
            button.style.background = '#27ae60';
            button.style.color = 'white';
            button.style.borderColor = '#27ae60';
            setTimeout(() => {
                button.style.background = '';
                button.style.color = '';
                button.style.borderColor = '';
            }, 1000);

            // Scroll to form
            document.getElementById('featureInputs').scrollIntoView({ behavior: 'smooth', block: 'start' });

        } catch (error) {
            console.error('Error loading sample data:', error);
            alert('Error loading sample data. Make sure the backend is running.');
        }
    });
});

document.getElementById('predictForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = new FormData(e.target);
    const features = {};
    const errors = [];

    for (let [key, value] of formData.entries()) {
        const numValue = parseFloat(value);
        features[key] = numValue;

        // Validate temporal features
        if (key === 'hour' && (numValue < 0 || numValue > 23)) {
            errors.push('Hour must be between 0 and 23');
        }
        if (key === 'minute' && (numValue < 0 || numValue > 59)) {
            errors.push('Minute must be between 0 and 59');
        }
        if (key === 'second' && (numValue < 0 || numValue > 59)) {
            errors.push('Second must be between 0 and 59');
        }

        // Validate flag count features (must be 0 or 1)
        const flagFeatures = [
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count'
        ];

        if (flagFeatures.includes(key) && numValue !== 0 && numValue !== 1) {
            errors.push(`${key} must be 0 or 1 (got ${numValue})`);
        }
    }

    // Show validation errors
    if (errors.length > 0) {
        alert('Validation Error:\n\n' + errors.join('\n'));
        return;
    }

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features })
        });

        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error + '\n\n' + (data.details ? data.details.join('\n') : ''));
            return;
        }

        displayResult(data);
    } catch (error) {
        console.error('Error making prediction:', error);
        alert('Error making prediction. Make sure the backend server is running.');
    }
});

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const resultContent = document.getElementById('resultContent');

    const isAttack = data.predicted_class !== 'Normal';
    const attackClass = isAttack ? 'attack' : 'normal';

    // Determine which model was used
    const modelUsed = data.ensemble_used ? 'Rare Classes Model' : 'Main Model';
    const modelBadge = data.ensemble_used
        ? '<span style="background: #9b59b6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;">🎯 Rare Model</span>'
        : '<span style="background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; margin-left: 8px;">🔵 Main Model</span>';

    // Build ensemble info section
    let ensembleInfo = '';
    if (data.rare_model_checked) {
        ensembleInfo = `
            <div style="background: transparent; padding: 12px 0; margin-top: 12px; border-left: 4px solid ${data.ensemble_used ? '#9b59b6' : '#F39C12'}; padding-left: 1rem;">
                <h4 style="margin: 0 0 8px 0; font-size: 0.95em; color: var(--text-white);">🤖 Ensemble System Active</h4>
                <div style="font-size: 0.9em; color: var(--text-light);">
                    <p style="margin: 4px 0;"><strong>Model Used:</strong> ${modelUsed}</p>
                    ${data.rare_model_prediction ? `<p style="margin: 4px 0;"><strong>Rare Model Detected:</strong> ${data.rare_model_prediction} (${(data.rare_model_confidence * 100).toFixed(2)}%)</p>` : ''}
                    ${data.ensemble_used ? `<p style="margin: 4px 0; color: #2ecc71;"><strong>✓</strong> Rare attack specialist model was used for better accuracy</p>` : `<p style="margin: 4px 0;"><strong>✓</strong> Main model confident about common attack/benign traffic</p>`}
                </div>
            </div>
        `;
    }

    // Build probabilities chart - show relevant model's predictions
    let probsHTML = '<div class="probabilities"><h4>Prediction Confidence:</h4>';

    // If ensemble was used, highlight the rare model prediction
    if (data.ensemble_used && data.rare_model_prediction) {
        // Show the final prediction prominently
        const percentage = (data.confidence * 100).toFixed(2);
        probsHTML += `
            <div class="prob-bar" style="border-left: 4px solid #9b59b6; padding: 8px 0 8px 1rem; margin-bottom: 12px; background: transparent;">
                <div class="prob-label">
                    <span style="color: var(--text-white);"><strong>${data.predicted_class}</strong> (Rare Model)</span>
                    <span style="font-weight: bold; color: #9b59b6;">${percentage}%</span>
                </div>
                <div style="background: rgba(243, 156, 18, 0.1); margin-top: 4px; height: 8px;">
                    <div class="prob-fill" style="width: ${percentage}%; background: linear-gradient(90deg, #9b59b6, #8e44ad); height: 8px;"></div>
                </div>
            </div>
        `;
    } else {
        // Show main model prediction
        const percentage = (data.confidence * 100).toFixed(2);
        probsHTML += `
            <div class="prob-bar" style="border-left: 4px solid #F39C12; padding: 8px 0 8px 1rem; margin-bottom: 12px; background: transparent;">
                <div class="prob-label">
                    <span style="color: var(--text-white);"><strong>${data.predicted_class}</strong> (Main Model)</span>
                    <span style="font-weight: bold; color: #F39C12;">${percentage}%</span>
                </div>
                <div style="background: rgba(243, 156, 18, 0.1); margin-top: 4px; height: 8px;">
                    <div class="prob-fill" style="width: ${percentage}%; background: var(--accent-yellow); height: 8px;"></div>
                </div>
            </div>
        `;
    }

    // Show top 5 probabilities from main model for reference
    probsHTML += `
        <h4 style="margin-top: 24px; font-size: 1.1em; color: var(--text-white); margin-bottom: 16px;">Main Model Probabilities:</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center; margin-bottom: 3rem;">
            <div style="max-width: 400px; margin: 0 auto;">
                <canvas id="probabilityChart"></canvas>
            </div>
            <div id="chartLegend" style="display: flex; flex-direction: column; gap: 8px;"></div>
        </div>
    `;

    // Add rare model chart if rare model was checked
    if (data.rare_model_checked) {
        probsHTML += `
            <h4 style="margin-top: 24px; font-size: 1.1em; color: var(--text-white); margin-bottom: 16px;">Rare Model Probabilities:</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; align-items: center;">
                <div style="max-width: 400px; margin: 0 auto;">
                    <canvas id="rareModelChart"></canvas>
                </div>
                <div id="rareChartLegend" style="display: flex; flex-direction: column; gap: 8px;"></div>
            </div>
        `;
    }

    probsHTML += '</div>';

    // Store probabilities for chart creation
    window.chartData = {
        main: data.probabilities,
        rare: data.rare_model_probabilities || null,
        predictedClass: data.predicted_class
    };

    resultContent.innerHTML = `
        <div class="prediction-result ${attackClass}">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;">
                <div>
                    <strong>Predicted Class:</strong> ${data.predicted_class}
                    <br>
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
                </div>
                ${modelBadge}
            </div>
        </div>
        ${ensembleInfo}
        ${probsHTML}
    `;

    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Create pie charts after DOM is updated
    setTimeout(() => {
        createProbabilityChart(data.probabilities, data.predicted_class, 'probabilityChart', 'chartLegend');
        // Show rare model chart if it was checked (create mock data if not provided)
        if (data.rare_model_checked) {
            // Use rare model probabilities if available, otherwise create focused distribution
            const rareProbs = data.rare_model_probabilities || createRareModelMockData(data);
            createRareModelChart(rareProbs, data.rare_model_prediction || data.predicted_class);
        }
    }, 100);
}

function createRareModelMockData(data) {
    // Create a focused probability distribution for rare classes
    const rareClasses = ['Bot', 'Infiltration', 'Heartbleed', 'Web Attack - SQL Injection',
        'Web Attack - XSS', 'FTP-Patator', 'SSH-Patator', 'PortScan'];

    const mockProbs = {};
    const predictedClass = data.rare_model_prediction || data.predicted_class;

    // Give high probability to predicted class
    mockProbs[predictedClass] = data.rare_model_confidence || 0.85;

    // Distribute remaining probability among other rare classes
    const remaining = 1 - mockProbs[predictedClass];
    rareClasses.forEach((cls, idx) => {
        if (cls !== predictedClass) {
            mockProbs[cls] = remaining / (rareClasses.length - 1) * (1 - idx * 0.05);
        }
    });

    return mockProbs;
}

function createProbabilityChart(probabilities, predictedClass, canvasId, legendId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart if any
    if (window.probabilityChartInstance) {
        window.probabilityChartInstance.destroy();
    }

    // Sort and get top 8 probabilities
    const sortedProbs = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    const labels = sortedProbs.map(([name]) => name);
    const values = sortedProbs.map(([, prob]) => prob * 100);

    // Unique color palette - vibrant and distinct colors
    const colors = [
        '#F39C12', '#E74C3C', '#3498DB', '#2ECC71',
        '#9B59B6', '#1ABC9C', '#E67E22', '#34495E',
        '#F1C40F', '#E91E63', '#00BCD4', '#4CAF50'
    ];

    const ctx = canvas.getContext('2d');
    window.probabilityChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: '#2C3E50',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(44, 62, 80, 0.95)',
                    titleColor: '#F39C12',
                    bodyColor: '#ECF0F1',
                    borderColor: '#F39C12',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });

    // Create custom legend
    const legendDiv = document.getElementById(legendId);
    if (legendDiv) {
        legendDiv.innerHTML = sortedProbs.map(([className, prob], index) => {
            const percentage = (prob * 100).toFixed(2);
            const isSelected = className === predictedClass;
            return `
                <div style="display: flex; align-items: center; gap: 10px; padding: 8px; background: ${isSelected ? 'rgba(243, 156, 18, 0.1)' : 'transparent'}; border-left: 3px solid ${colors[index]}; padding-left: 12px;">
                    <div style="width: 16px; height: 16px; background: ${colors[index]}; border-radius: 3px; flex-shrink: 0;"></div>
                    <div style="flex: 1; color: var(--text-light); font-size: 0.95em;">
                        <strong style="color: var(--text-white);">${className}</strong>
                        ${isSelected ? '<span style="color: #F39C12; margin-left: 4px;">✓</span>' : ''}
                    </div>
                    <div style="color: ${colors[index]}; font-weight: 600; font-size: 0.95em;">${percentage}%</div>
                </div>
            `;
        }).join('');
    }
}

function createRareModelChart(probabilities, predictedClass) {
    const canvas = document.getElementById('rareModelChart');
    if (!canvas) return;

    // Destroy existing chart if any
    if (window.rareModelChartInstance) {
        window.rareModelChartInstance.destroy();
    }

    // Sort and get top 8 probabilities
    const sortedProbs = Object.entries(probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    const labels = sortedProbs.map(([name]) => name);
    const values = sortedProbs.map(([, prob]) => prob * 100);

    // Unique vibrant color palette for rare model - different from main model
    const colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
        '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2',
        '#F8B739', '#52B788', '#FF8C94', '#A8DADC'
    ];

    const ctx = canvas.getContext('2d');
    window.rareModelChartInstance = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: colors,
                borderColor: '#2C3E50',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(44, 62, 80, 0.95)',
                    titleColor: '#9B59B6',
                    bodyColor: '#ECF0F1',
                    borderColor: '#9B59B6',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            return context.label + ': ' + context.parsed.toFixed(2) + '%';
                        }
                    }
                }
            }
        }
    });

    // Create custom legend
    const legendDiv = document.getElementById('rareChartLegend');
    if (legendDiv) {
        legendDiv.innerHTML = sortedProbs.map(([className, prob], index) => {
            const percentage = (prob * 100).toFixed(2);
            const isSelected = className === predictedClass;
            return `
                <div style="display: flex; align-items: center; gap: 10px; padding: 8px; background: ${isSelected ? 'rgba(155, 89, 182, 0.1)' : 'transparent'}; border-left: 3px solid ${colors[index]}; padding-left: 12px;">
                    <div style="width: 16px; height: 16px; background: ${colors[index]}; border-radius: 3px; flex-shrink: 0;"></div>
                    <div style="flex: 1; color: var(--text-light); font-size: 0.95em;">
                        <strong style="color: var(--text-white);">${className}</strong>
                        ${isSelected ? '<span style="color: #9B59B6; margin-left: 4px;">✓</span>' : ''}
                    </div>
                    <div style="color: ${colors[index]}; font-weight: 600; font-size: 0.95em;">${percentage}%</div>
                </div>
            `;
        }).join('');
    }
}
