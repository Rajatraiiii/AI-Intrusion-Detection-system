// Main JavaScript for AI-Powered Intrusion Detection System

// Tab switching
function showTab(tabName, event) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab with animation
    const targetTab = document.getElementById(tabName);
    targetTab.classList.add('active');
    
    // Activate corresponding button
    if (event && event.target) {
        event.target.closest('.tab-btn').classList.add('active');
    } else {
        document.querySelectorAll('.tab-btn').forEach(btn => {
            if (btn.textContent.includes(tabName.charAt(0).toUpperCase() + tabName.slice(1))) {
                btn.classList.add('active');
            }
        });
    }
}

// File upload handling
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('dataset');
    const fileUploadBox = document.getElementById('fileUploadBox');
    const fileInfo = document.getElementById('fileInfo');
    
    if (fileInput && fileUploadBox) {
        // Click to upload
        fileUploadBox.addEventListener('click', () => {
            fileInput.click();
        });
        
        // Drag and drop
        fileUploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadBox.style.borderColor = 'var(--secondary)';
            fileUploadBox.style.transform = 'scale(1.02)';
        });
        
        fileUploadBox.addEventListener('dragleave', () => {
            fileUploadBox.style.borderColor = 'var(--primary)';
            fileUploadBox.style.transform = 'scale(1)';
        });
        
        fileUploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadBox.style.borderColor = 'var(--primary)';
            fileUploadBox.style.transform = 'scale(1)';
            
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                updateFileInfo(e.dataTransfer.files[0]);
            }
        });
        
        // File selected
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                updateFileInfo(e.target.files[0]);
            }
        });
    }
    
    function updateFileInfo(file) {
        if (file) {
            const fileSize = (file.size / 1024 / 1024).toFixed(2);
            fileInfo.innerHTML = `
                <i class="fas fa-check-circle"></i>
                <strong>${file.name}</strong> (${fileSize} MB)
            `;
            fileInfo.classList.add('show');
        }
    }
});

// Train Models Form
const trainFormEl = document.getElementById('trainForm');
if (trainFormEl) trainFormEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('dataset');
    
    if (!fileInput.files[0]) {
        showNotification('Please select a file first', 'error');
        return;
    }
    
    formData.append('file', fileInput.files[0]);
    
    const statusDiv = document.getElementById('trainStatus');
    const resultsDiv = document.getElementById('trainResults');
    const progressContainer = document.getElementById('trainProgress');
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    
    // Show progress
    progressContainer.style.display = 'block';
    statusDiv.className = 'status info';
    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training models... This may take several minutes.';
    resultsDiv.innerHTML = '';
    
    // Animate progress (simulated)
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressFill.style.width = progress + '%';
        progressText.textContent = `Training in progress... ${Math.floor(progress)}%`;
    }, 1000);
    
    try {
        const response = await fetch('/api/train', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressText.textContent = 'Training complete!';
        
        const data = await response.json();
        
        if (response.ok) {
            setTimeout(() => {
                progressContainer.style.display = 'none';
            }, 1000);
            
            statusDiv.className = 'status success';
            statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> Models trained successfully!';
            
            // Display results with animations
            let resultsHTML = '<div class="results-grid">';
            
            const modelIcons = {
                'logistic_regression': 'fas fa-chart-line',
                'random_forest': 'fas fa-tree',
                'neural_network': 'fas fa-brain'
            };
            
            let delay = 0;
            for (const [modelName, metrics] of Object.entries(data.results)) {
                const icon = modelIcons[modelName] || 'fas fa-chart-bar';
                resultsHTML += `
                    <div class="result-card" style="animation-delay: ${delay}s">
                        <h3>
                            <i class="${icon}"></i>
                            ${modelName.replace('_', ' ').toUpperCase()}
                        </h3>
                        <div class="metric">
                            <span class="metric-label">Accuracy:</span>
                            <span class="metric-value">${(metrics.accuracy * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Precision:</span>
                            <span class="metric-value">${(metrics.precision * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">Recall:</span>
                            <span class="metric-value">${(metrics.recall * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">F1-Score:</span>
                            <span class="metric-value">${(metrics.f1_score * 100).toFixed(2)}%</span>
                        </div>
                        <div class="metric">
                            <span class="metric-label">ROC AUC:</span>
                            <span class="metric-value">${(metrics.roc_auc * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                `;
                delay += 0.1;
            }
            
            resultsHTML += '</div>';
            resultsDiv.innerHTML = resultsHTML;
            
            showNotification('Models trained successfully!', 'success');
        } else {
            progressContainer.style.display = 'none';
            statusDiv.className = 'status error';
            statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${data.error}`;
            showNotification(data.error, 'error');
        }
    } catch (error) {
        clearInterval(progressInterval);
        progressContainer.style.display = 'none';
        statusDiv.className = 'status error';
        statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${error.message}`;
        showNotification(error.message, 'error');
    }
});

// Predict Form
const predictFormEl = document.getElementById('predictForm');
if (predictFormEl) predictFormEl.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {};
    
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });
    
    const statusDiv = document.getElementById('predictStatus');
    const resultsDiv = document.getElementById('predictResults');
    
    statusDiv.className = 'status info';
    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing traffic...';
    resultsDiv.innerHTML = '';
    
    // Add loading animation
    resultsDiv.innerHTML = '<div style="text-align: center; padding: 40px;"><i class="fas fa-cog fa-spin" style="font-size: 3rem; color: var(--primary);"></i></div>';
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            statusDiv.className = 'status success';
            statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> Prediction complete!';
            
            // Display results with animations
            let resultsHTML = '<div class="results-grid">';
            
            const finalPrediction = result.final_prediction;
            const predictionClass = finalPrediction === 'Attack' ? 'prediction-attack' : 'prediction-normal';
            const predictionIcon = finalPrediction === 'Attack' ? 'fa-exclamation-triangle' : 'fa-shield-check';
            
            resultsHTML += `
                <div class="result-card" style="grid-column: 1 / -1; background: linear-gradient(135deg, ${finalPrediction === 'Attack' ? '#fee2e2' : '#d1fae5'}, #ffffff);">
                    <h3>
                        <i class="fas ${predictionIcon}"></i>
                        Final Prediction
                    </h3>
                    <div style="text-align: center; margin: 20px 0;">
                        <span class="prediction-badge ${predictionClass}">
                            <i class="fas ${predictionIcon}"></i>
                            ${finalPrediction}
                        </span>
                        <p style="margin-top: 20px; font-size: 1.1rem; color: var(--gray);">
                            Confidence: <strong style="color: var(--dark);">${(result.confidence * 100).toFixed(2)}%</strong>
                        </p>
                    </div>
                </div>
            `;
            
            const modelIcons = {
                'logistic_regression': 'fa-chart-line',
                'random_forest': 'fa-tree',
                'neural_network': 'fa-brain'
            };
            
            let delay = 0.1;
            for (const [modelName, prediction] of Object.entries(result.predictions)) {
                if (prediction.error) {
                    resultsHTML += `
                        <div class="result-card">
                            <h3><i class="fas fa-exclamation-circle"></i> ${modelName.replace('_', ' ').toUpperCase()}</h3>
                            <p style="color: var(--danger); padding: 15px; background: #fee2e2; border-radius: 8px;">
                                <i class="fas fa-times-circle"></i> ${prediction.error}
                            </p>
                        </div>
                    `;
                } else {
                    const predClass = prediction.prediction === 'Attack' ? 'prediction-attack' : 'prediction-normal';
                    const icon = modelIcons[modelName] || 'fa-chart-bar';
                    resultsHTML += `
                        <div class="result-card" style="animation-delay: ${delay}s">
                            <h3>
                                <i class="fas ${icon}"></i>
                                ${modelName.replace('_', ' ').toUpperCase()}
                            </h3>
                            <div style="text-align: center; margin: 15px 0;">
                                <span class="prediction-badge ${predClass}">${prediction.prediction}</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Confidence:</span>
                                <span class="metric-value">${(prediction.confidence * 100).toFixed(2)}%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Probability:</span>
                                <span class="metric-value">${(prediction.probability * 100).toFixed(2)}%</span>
                            </div>
                        </div>
                    `;
                    delay += 0.1;
                }
            }
            
            resultsHTML += '</div>';
            resultsDiv.innerHTML = resultsHTML;
            
            showNotification(`Prediction: ${finalPrediction}`, finalPrediction === 'Attack' ? 'warning' : 'success');
        } else {
            statusDiv.className = 'status error';
            statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${result.error}`;
            showNotification(result.error, 'error');
        }
    } catch (error) {
        statusDiv.className = 'status error';
        statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${error.message}`;
        showNotification(error.message, 'error');
    }
});

// Fill sample data
function fillSampleData() {
    const sampleData = {
        flow_duration: 123456,
        total_fwd_packets: 100,
        total_backward_packets: 50,
        total_length_fwd_packets: 50000,
        total_length_bwd_packets: 25000,
        fwd_packet_length_max: 1500,
        fwd_packet_length_min: 60,
        bwd_packet_length_max: 1500,
        bwd_packet_length_min: 60,
        flow_bytes_s: 500000,
        flow_packets_s: 1000,
        packet_length_mean: 750,
        packet_length_std: 300,
        source_port: 12345,
        destination_port: 80,
        protocol: 6
    };
    
    let filled = 0;
    for (const [key, value] of Object.entries(sampleData)) {
        const input = document.querySelector(`input[name="${key}"]`);
        if (input) {
            input.value = value;
            input.style.background = '#e0e7ff';
            setTimeout(() => {
                input.style.transition = 'background 0.5s';
                input.style.background = '';
            }, 300);
            filled++;
        }
    }
    
    if (filled > 0) {
        showNotification('Sample data filled successfully!', 'success');
    }
}

// Clear form
function clearForm() {
    document.querySelectorAll('#predictForm input').forEach(input => {
        if (input.type === 'number') {
            input.value = 0;
            input.style.background = '#fee2e2';
            setTimeout(() => {
                input.style.transition = 'background 0.5s';
                input.style.background = '';
            }, 300);
        }
    });
    showNotification('Form cleared', 'info');
}

// Notification system
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

// Load evaluation results
async function loadEvaluation() {
    const statusDiv = document.getElementById('evalStatus');
    const resultsDiv = document.getElementById('evalResults');
    
    statusDiv.className = 'status info';
    statusDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading evaluation results...';
    resultsDiv.innerHTML = '<div style="text-align: center; padding: 40px;"><i class="fas fa-cog fa-spin" style="font-size: 3rem; color: var(--primary);"></i></div>';
    
    try {
        const response = await fetch('/api/evaluation');
        const data = await response.json();
        
        if (response.ok && Object.keys(data).length > 0) {
            statusDiv.className = 'status success';
            statusDiv.innerHTML = '<i class="fas fa-check-circle"></i> Evaluation results loaded!';
            
            // Display results table with enhanced styling
            let resultsHTML = `
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th><i class="fas fa-brain"></i> Model</th>
                                <th><i class="fas fa-chart-line"></i> Accuracy</th>
                                <th><i class="fas fa-bullseye"></i> Precision</th>
                                <th><i class="fas fa-redo"></i> Recall</th>
                                <th><i class="fas fa-tachometer-alt"></i> F1-Score</th>
                                <th><i class="fas fa-chart-area"></i> ROC AUC</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            for (const [modelName, metrics] of Object.entries(data)) {
                const modelIcons = {
                    'logistic_regression': 'fa-chart-line',
                    'random_forest': 'fa-tree',
                    'neural_network': 'fa-brain'
                };
                const icon = modelIcons[modelName] || 'fa-chart-bar';
                
                resultsHTML += `
                    <tr>
                        <td>
                            <strong>
                                <i class="fas ${icon}"></i>
                                ${modelName.replace('_', ' ').toUpperCase()}
                            </strong>
                        </td>
                        <td><span class="metric-badge">${(metrics.accuracy * 100).toFixed(2)}%</span></td>
                        <td><span class="metric-badge">${(metrics.precision * 100).toFixed(2)}%</span></td>
                        <td><span class="metric-badge">${(metrics.recall * 100).toFixed(2)}%</span></td>
                        <td><span class="metric-badge">${(metrics.f1_score * 100).toFixed(2)}%</span></td>
                        <td><span class="metric-badge">${(metrics.roc_auc * 100).toFixed(2)}%</span></td>
                    </tr>
                `;
            }
            
            resultsHTML += '</tbody></table></div>';
            
            // Add images if available
            const imageNames = ['performance_comparison.png', 'roc_curves.png'];
            for (const imageName of imageNames) {
                resultsHTML += `
                    <div class="image-container">
                        <img src="/static/images/${imageName}?t=${Date.now()}" alt="${imageName}" onerror="this.style.display='none'" loading="lazy">
                    </div>
                `;
            }
            
            resultsDiv.innerHTML = resultsHTML;
            showNotification('Evaluation results loaded successfully!', 'success');
        } else {
            statusDiv.className = 'status error';
            statusDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i> No evaluation results found. Please train models first.';
            resultsDiv.innerHTML = '';
            showNotification('No evaluation results found', 'warning');
        }
    } catch (error) {
        statusDiv.className = 'status error';
        statusDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> Error: ${error.message}`;
        resultsDiv.innerHTML = '';
        showNotification(error.message, 'error');
    }
}

// Check model status on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/models/status');
        const data = await response.json();
        
        if (data.loaded) {
            console.log('Models loaded:', data.models);
        }
    } catch (error) {
        console.error('Error checking model status:', error);
    }
});

