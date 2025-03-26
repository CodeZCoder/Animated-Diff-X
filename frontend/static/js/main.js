// AnimateDiff GUI - Main JavaScript

// Socket.io connection
let socket;
let currentGenerationId = null;

// DOM elements
const statusBadge = document.getElementById('status-badge');
const generateBtn = document.getElementById('generate-btn');
const cancelBtn = document.getElementById('cancel-btn');
const progressContainer = document.getElementById('progress-container');
// Progress bars
const stepsProgressBar = document.getElementById('steps-progress-bar');
const framesProgressBar = document.getElementById('frames-progress-bar');
const fpsProgressBar = document.getElementById('fps-progress-bar');
const totalProgressBar = document.getElementById('total-progress-bar');
// Progress text elements
const stepsProgressText = document.getElementById('steps-progress-text');
const framesProgressText = document.getElementById('frames-progress-text');
const fpsProgressText = document.getElementById('fps-progress-text');
const totalProgressText = document.getElementById('total-progress-text');
// Other elements
const previewImage = document.getElementById('preview-image');
const placeholder = document.getElementById('placeholder');
const generationInfo = document.getElementById('generation-info');
const currentStepEl = document.getElementById('current-step');
const totalStepsEl = document.getElementById('total-steps');
const generationIdEl = document.getElementById('generation-id');
const resultVideo = document.getElementById('result-video');
const videoContainer = document.getElementById('video-container');
const noResults = document.getElementById('no-results');
const downloadLink = document.getElementById('download-link');
const errorModal = new bootstrap.Modal(document.getElementById('error-modal'));
const errorMessage = document.getElementById('error-message');

// Form elements
const generationForm = document.getElementById('generation-form');
const promptInput = document.getElementById('prompt');
const negativePromptInput = document.getElementById('negative-prompt');
const sdModelSelect = document.getElementById('sd-model');
const motionModuleSelect = document.getElementById('motion-module');
const motionLoraSelect = document.getElementById('motion-lora');
const motionLoraStrengthInput = document.getElementById('motion-lora-strength');
const loraStrengthValue = document.getElementById('lora-strength-value');
const numFramesInput = document.getElementById('num-frames');
const fpsInput = document.getElementById('fps');
const widthInput = document.getElementById('width');
const heightInput = document.getElementById('height');
const guidanceScaleInput = document.getElementById('guidance-scale');
const guidanceScaleValue = document.getElementById('guidance-scale-value');
const numInferenceStepsInput = document.getElementById('num-inference-steps');
const seedInput = document.getElementById('seed');
const useImageToVideoCheckbox = document.getElementById('use-image-to-video');
const imageUploadContainer = document.getElementById('image-upload-container');
const inputImageInput = document.getElementById('input-image');
const imagePreview = document.getElementById('image-preview');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initSocketConnection();
    loadModels();
    setupEventListeners();
    updateUIState('ready');
});

// Initialize Socket.io connection
function initSocketConnection() {
    // Connect to the server
    socket = io();
    
    // Connection events
    socket.on('connect', () => {
        console.log('Connected to server');
        updateUIState('ready');
    });
    
    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateUIState('disconnected');
    });
    
    // Generation events
    socket.on('generation_started', (data) => {
        console.log('Generation started:', data);
        currentGenerationId = data.generation_id;
        generationIdEl.textContent = data.generation_id;
        totalStepsEl.textContent = data.total_steps;
        updateUIState('generating');
    });
    
    socket.on('preview_update', (data) => {
        console.log('Preview update:', data);
        if (data.generation_id === currentGenerationId) {
            // Update preview image
            previewImage.src = data.preview_image;
            previewImage.classList.remove('d-none');
            placeholder.classList.add('d-none');
            
            // Update steps progress
            const stepsProgress = (data.step / data.total_steps) * 100;
            stepsProgressBar.style.width = `${stepsProgress}%`;
            stepsProgressBar.setAttribute('aria-valuenow', stepsProgress);
            stepsProgressText.textContent = `${Math.round(stepsProgress)}%`;
            
            // Update frames progress
            const framesProgress = data.frames_progress || 0;
            framesProgressBar.style.width = `${framesProgress}%`;
            framesProgressBar.setAttribute('aria-valuenow', framesProgress);
            framesProgressText.textContent = `${Math.round(framesProgress)}%`;
            
            // Update FPS progress
            const fpsProgress = data.fps_progress || 0;
            fpsProgressBar.style.width = `${fpsProgress}%`;
            fpsProgressBar.setAttribute('aria-valuenow', fpsProgress);
            fpsProgressText.textContent = `${Math.round(fpsProgress)}%`;
            
            // Update total progress
            const totalProgress = data.total_progress || data.progress || stepsProgress;
            totalProgressBar.style.width = `${totalProgress}%`;
            totalProgressBar.setAttribute('aria-valuenow', totalProgress);
            totalProgressText.textContent = `${Math.round(totalProgress)}%`;
            
            // Update step info
            currentStepEl.textContent = data.step;
            totalStepsEl.textContent = data.total_steps;
        }
    });
    
    socket.on('generation_completed', (data) => {
        console.log('Generation completed:', data);
        if (data.generation_id === currentGenerationId) {
            updateUIState('completed');
            
            // Display the result video
            resultVideo.src = data.output_url;
            videoContainer.classList.remove('d-none');
            noResults.classList.add('d-none');
            
            // Set download link
            downloadLink.href = data.output_url;
            downloadLink.download = data.output_url.split('/').pop();
        }
    });
    
    socket.on('generation_error', (data) => {
        console.error('Generation error:', data);
        if (data.generation_id === currentGenerationId || !data.generation_id) {
            updateUIState('error');
            showError(data.error || 'An unknown error occurred');
        }
    });
    
    socket.on('generation_cancelled', (data) => {
        console.log('Generation cancelled:', data);
        if (data.generation_id === currentGenerationId) {
            updateUIState('cancelled');
        }
    });
}

// Load available models from the server
function loadModels() {
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            // Populate SD models
            populateSelect(sdModelSelect, data.stable_diffusion);
            
            // Populate motion modules
            populateSelect(motionModuleSelect, data.motion_module);
            
            // Populate motion LoRAs
            populateSelect(motionLoraSelect, data.motion_lora, true);
        })
        .catch(error => {
            console.error('Error loading models:', error);
            showError('Failed to load models. Please check if the server is running.');
        });
}

// Populate a select element with options
function populateSelect(selectElement, options, includeNone = false) {
    // Clear existing options
    selectElement.innerHTML = '';
    
    // Add "None" option if requested
    if (includeNone) {
        const noneOption = document.createElement('option');
        noneOption.value = 'none';
        noneOption.textContent = 'None';
        selectElement.appendChild(noneOption);
    }
    
    // Add options from the server
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        selectElement.appendChild(optionElement);
    });
}

// Set up event listeners
function setupEventListeners() {
    // Form submission
    generationForm.addEventListener('submit', (e) => {
        e.preventDefault();
        startGeneration();
    });
    
    // Cancel button
    cancelBtn.addEventListener('click', () => {
        cancelGeneration();
    });
    
    // Image to video toggle
    useImageToVideoCheckbox.addEventListener('change', () => {
        imageUploadContainer.style.display = useImageToVideoCheckbox.checked ? 'block' : 'none';
    });
    
    // Input image preview
    inputImageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imagePreview.src = e.target.result;
                imagePreview.classList.remove('d-none');
            };
            reader.readAsDataURL(file);
        }
    });
    
    // Update slider value displays
    motionLoraStrengthInput.addEventListener('input', () => {
        loraStrengthValue.textContent = motionLoraStrengthInput.value;
    });
    
    guidanceScaleInput.addEventListener('input', () => {
        guidanceScaleValue.textContent = guidanceScaleInput.value;
    });
}

// Start the generation process
function startGeneration() {
    // Prepare the data
    const data = {
        prompt: promptInput.value,
        negative_prompt: negativePromptInput.value,
        sd_model: sdModelSelect.value,
        motion_module: motionModuleSelect.value,
        motion_lora: motionLoraSelect.value === 'none' ? null : motionLoraSelect.value,
        motion_lora_strength: parseFloat(motionLoraStrengthInput.value),
        num_frames: parseInt(numFramesInput.value),
        fps: parseInt(fpsInput.value),
        guidance_scale: parseFloat(guidanceScaleInput.value),
        num_inference_steps: parseInt(numInferenceStepsInput.value),
        seed: parseInt(seedInput.value),
        width: parseInt(widthInput.value),
        height: parseInt(heightInput.value)
    };
    
    // Add input image if using image to video
    if (useImageToVideoCheckbox.checked && inputImageInput.files.length > 0) {
        data.input_image = imagePreview.src;
    }
    
    // Update UI
    updateUIState('starting');
    
    // Reset preview
    previewImage.classList.add('d-none');
    placeholder.classList.remove('d-none');
    
    // Send the request
    socket.emit('start_generation', data, (response) => {
        if (response && response.status === 'error') {
            updateUIState('error');
            showError(response.error || 'Failed to start generation');
        }
    });
}

// Cancel the current generation
function cancelGeneration() {
    if (currentGenerationId) {
        socket.emit('cancel_generation', { generation_id: currentGenerationId });
        updateUIState('cancelling');
    }
}

// Update the UI state
function updateUIState(state) {
    switch (state) {
        case 'ready':
            statusBadge.textContent = 'Ready';
            statusBadge.className = 'badge bg-success';
            generateBtn.disabled = false;
            cancelBtn.classList.add('d-none');
            progressContainer.style.display = 'none';
            generationInfo.style.display = 'none';
            break;
            
        case 'starting':
            statusBadge.textContent = 'Starting...';
            statusBadge.className = 'badge bg-warning';
            generateBtn.disabled = true;
            break;
            
        case 'generating':
            statusBadge.textContent = 'Generating';
            statusBadge.className = 'badge bg-primary';
            generateBtn.disabled = true;
            cancelBtn.classList.remove('d-none');
            progressContainer.style.display = 'block';
            generationInfo.style.display = 'block';
            break;
            
        case 'cancelling':
            statusBadge.textContent = 'Cancelling...';
            statusBadge.className = 'badge bg-warning';
            generateBtn.disabled = true;
            cancelBtn.disabled = true;
            break;
            
        case 'cancelled':
            statusBadge.textContent = 'Cancelled';
            statusBadge.className = 'badge bg-secondary';
            generateBtn.disabled = false;
            cancelBtn.classList.add('d-none');
            progressContainer.style.display = 'none';
            setTimeout(() => {
                if (statusBadge.textContent === 'Cancelled') {
                    updateUIState('ready');
                }
            }, 3000);
            break;
            
        case 'completed':
            statusBadge.textContent = 'Completed';
            statusBadge.className = 'badge bg-success';
            generateBtn.disabled = false;
            cancelBtn.classList.add('d-none');
            progressContainer.style.display = 'none';
            break;
            
        case 'error':
            statusBadge.textContent = 'Error';
            statusBadge.className = 'badge bg-danger';
            generateBtn.disabled = false;
            cancelBtn.classList.add('d-none');
            progressContainer.style.display = 'none';
            break;
            
        case 'disconnected':
            statusBadge.textContent = 'Disconnected';
            statusBadge.className = 'badge bg-danger';
            generateBtn.disabled = true;
            cancelBtn.classList.add('d-none');
            progressContainer.style.display = 'none';
            showError('Disconnected from server. Please refresh the page.');
            break;
    }
}

// Show an error message
function showError(message) {
    errorMessage.textContent = message;
    errorModal.show();
}
