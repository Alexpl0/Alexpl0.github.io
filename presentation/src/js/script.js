// Enhanced JavaScript functionality for Technical CNN 1D Presentation
// Handles slide navigation, progress bar updates, keyboard controls, and advanced features

let currentSlide = 1;
// Updated total number of slides to reflect the enhanced technical version
let totalSlides = 40; // Increased from 34 to 40 for new technical slides

// Slide management functions
function showSlide(n) {
    // Hide all slides
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => slide.style.display = 'none');
    
    // Show current slide
    const current = document.querySelector(`[data-slide="${n}"]`);
    if (current) {
        current.style.display = 'flex'; // Use flex to maintain layout
        // Trigger animation
        current.style.animation = 'none';
        current.offsetHeight; /* trigger reflow */
        current.style.animation = null; 
        
        // Update slide-specific features
        updateSlideFeatures(n);
    }
    
    // Update counter
    document.getElementById('slideCounter').textContent = `${n} / ${totalSlides}`;
    
    // Update progress bar
    const progress = (n / totalSlides) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
    
    // Update URL hash for bookmarking
    window.location.hash = `slide-${n}`;
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        showSlide(currentSlide);
        logSlideTransition('next', currentSlide);
    }
}

function previousSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        showSlide(currentSlide);
        logSlideTransition('previous', currentSlide);
    }
}

// Enhanced slide-specific features
function updateSlideFeatures(slideNumber) {
    // Re-render MathJax for mathematical equations
    if (typeof MathJax !== 'undefined') {
        MathJax.typesetPromise().catch((err) => console.log('MathJax error:', err));
    }
    
    // Initialize slide-specific animations
    initializeSlideAnimations(slideNumber);
    
    // Update audio controls if present
    updateAudioControls(slideNumber);
    
    // Initialize interactive elements
    initializeInteractiveElements(slideNumber);
}

// Initialize slide-specific animations
function initializeSlideAnimations(slideNumber) {
    const currentSlideElement = document.querySelector(`[data-slide="${slideNumber}"]`);
    if (!currentSlideElement) return;
    
    // Animate technical diagrams
    const diagrams = currentSlideElement.querySelectorAll('.technical-step-layout, .neural-architecture');
    diagrams.forEach((diagram, index) => {
        setTimeout(() => {
            diagram.style.opacity = '0';
            diagram.style.transform = 'translateY(20px)';
            diagram.style.transition = 'all 0.6s ease-out';
            
            requestAnimationFrame(() => {
                diagram.style.opacity = '1';
                diagram.style.transform = 'translateY(0)';
            });
        }, index * 200);
    });
    
    // Animate formula blocks
    const formulaBlocks = currentSlideElement.querySelectorAll('.formula-block');
    formulaBlocks.forEach((block, index) => {
        setTimeout(() => {
            block.style.opacity = '0';
            block.style.transform = 'scale(0.95)';
            block.style.transition = 'all 0.4s ease-out';
            
            requestAnimationFrame(() => {
                block.style.opacity = '1';
                block.style.transform = 'scale(1)';
            });
        }, 300 + index * 150);
    });
    
    // Animate layer boxes in neural architecture
    const layerBoxes = currentSlideElement.querySelectorAll('.layer-box');
    layerBoxes.forEach((box, index) => {
        setTimeout(() => {
            box.style.opacity = '0';
            box.style.transform = 'translateX(-30px)';
            box.style.transition = 'all 0.5s ease-out';
            
            requestAnimationFrame(() => {
                box.style.opacity = '1';
                box.style.transform = 'translateX(0)';
            });
        }, index * 100);
    });
}

// Handle audio controls
function updateAudioControls(slideNumber) {
    const currentSlideElement = document.querySelector(`[data-slide="${slideNumber}"]`);
    if (!currentSlideElement) return;
    
    const audioElements = currentSlideElement.querySelectorAll('audio');
    audioElements.forEach(audio => {
        // Pause any playing audio when leaving slide
        audio.pause();
        audio.currentTime = 0;
        
        // Add event listeners for accessibility
        audio.addEventListener('play', () => {
            logInteraction('audio_play', slideNumber);
        });
        
        audio.addEventListener('pause', () => {
            logInteraction('audio_pause', slideNumber);
        });
    });
}

// Initialize interactive elements
function initializeInteractiveElements(slideNumber) {
    const currentSlideElement = document.querySelector(`[data-slide="${slideNumber}"]`);
    if (!currentSlideElement) return;
    
    // Add hover effects to comparison tables
    const tables = currentSlideElement.querySelectorAll('.comparison-table, .mfcc-table, .cnn-table');
    tables.forEach(table => {
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach(row => {
            row.addEventListener('mouseenter', function() {
                this.style.backgroundColor = 'rgba(61, 157, 242, 0.1)';
                this.style.transform = 'translateX(2px)';
                this.style.transition = 'all 0.2s ease';
            });
            
            row.addEventListener('mouseleave', function() {
                this.style.backgroundColor = '';
                this.style.transform = 'translateX(0)';
            });
        });
    });
    
    // Add click-to-expand functionality for formula blocks
    const formulaBlocks = currentSlideElement.querySelectorAll('.formula-block');
    formulaBlocks.forEach(block => {
        block.addEventListener('click', function() {
            this.classList.toggle('expanded');
            if (this.classList.contains('expanded')) {
                this.style.transform = 'scale(1.02)';
                this.style.boxShadow = '0 8px 25px rgba(3, 18, 64, 0.2)';
                this.style.zIndex = '10';
            } else {
                this.style.transform = 'scale(1)';
                this.style.boxShadow = '';
                this.style.zIndex = '';
            }
        });
    });
    
    // Initialize cycle diagram interactions (slide 6)
    if (slideNumber === 6) {
        initializeCycleDiagram();
    }
    
    // Initialize neural network diagram interactions (slide 33)
    if (slideNumber === 33) {
        initializeNeuralNetworkDiagram();
    }
}

// Cycle diagram interactions
function initializeCycleDiagram() {
    const cycleSteps = document.querySelectorAll('.cycle-step');
    cycleSteps.forEach((step, index) => {
        step.addEventListener('click', function() {
            // Remove active class from all steps
            cycleSteps.forEach(s => s.classList.remove('active'));
            
            // Add active class to clicked step
            this.classList.add('active');
            
            // Show step details (could expand to show more info)
            showCycleStepDetails(index + 1);
        });
    });
}

// Neural network diagram interactions
function initializeNeuralNetworkDiagram() {
    const layerBoxes = document.querySelectorAll('.layer-box');
    layerBoxes.forEach((box, index) => {
        box.addEventListener('click', function() {
            // Toggle detailed view
            this.classList.toggle('detailed');
            
            if (this.classList.contains('detailed')) {
                showLayerDetails(index);
            } else {
                hideLayerDetails();
            }
        });
    });
}

// Show cycle step details
function showCycleStepDetails(stepNumber) {
    const stepNames = [
        'Adquisición de Datos',
        'Análisis y Preprocesamiento', 
        'Extracción de Características',
        'Entrenamiento del Modelo',
        'Evaluación y Resultados'
    ];
    
    const details = [
        'Descarga y organización de datasets de audio emocional (RAVDESS, TESS, MESD)',
        'Limpieza de datos, normalización de audio y preparación para extracción de características',
        'Cálculo de MFCCs, características espectrales y prosódicas usando Librosa',
        'Diseño de arquitectura CNN 1D, entrenamiento con ADAM y ajuste de hiperparámetros',
        'Validación del modelo, cálculo de métricas y análisis de resultados'
    ];
    
    console.log(`Paso ${stepNumber}: ${stepNames[stepNumber - 1]}`);
    console.log(`Detalle: ${details[stepNumber - 1]}`);
    
    // Could implement a tooltip or modal here
    logInteraction('cycle_step_selected', currentSlide, {step: stepNumber});
}

// Show layer details
function showLayerDetails(layerIndex) {
    const layerInfo = [
        'Capa de entrada que recibe el vector de 180 características extraídas del audio',
        'Primera capa convolucional con 256 filtros de tamaño 5, detecta patrones locales',
        'MaxPooling reduce dimensionalidad de 180 a 36, conserva características importantes',
        'Dropout desactiva 20% de neuronas aleatoriamente para prevenir overfitting',
        'Segunda capa convolucional con 128 filtros, detecta patrones de nivel superior',
        'Segundo MaxPooling reduce dimensionalidad de 36 a 7',
        'Segundo Dropout para regularización adicional',
        'Flatten convierte datos 2D a vector 1D de 896 elementos',
        'Capa densa final con 7 neuronas y Softmax para clasificación de emociones'
    ];
    
    console.log(`Capa ${layerIndex + 1}: ${layerInfo[layerIndex]}`);
    logInteraction('layer_details_viewed', currentSlide, {layer: layerIndex});
}

// Hide layer details
function hideLayerDetails() {
    console.log('Ocultando detalles de capa');
}

// Enhanced keyboard navigation
document.addEventListener('keydown', function(e) {
    switch(e.key) {
        case 'ArrowRight':
        case ' ':
            e.preventDefault();
            nextSlide();
            break;
        case 'ArrowLeft':
            e.preventDefault();
            previousSlide();
            break;
        case 'Home':
            e.preventDefault();
            currentSlide = 1;
            showSlide(currentSlide);
            break;
        case 'End':
            e.preventDefault();
            currentSlide = totalSlides;
            showSlide(currentSlide);
            break;
        case 'f':
        case 'F':
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                toggleFullscreen();
            }
            break;
        case 'p':
        case 'P':
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                togglePresentationMode();
            }
            break;
        case 'Escape':
            exitFullscreen();
            exitPresentationMode();
            break;
    }
});

// Fullscreen functionality
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(err => {
            console.log('Error attempting to enable fullscreen:', err);
        });
    } else {
        document.exitFullscreen();
    }
}

function exitFullscreen() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    }
}

// Presentation mode
let presentationMode = false;

function togglePresentationMode() {
    presentationMode = !presentationMode;
    
    if (presentationMode) {
        document.body.classList.add('presentation-mode');
        // Hide navigation and other UI elements
        document.querySelector('.navigation').style.display = 'none';
        document.querySelector('.slide-counter').style.display = 'none';
    } else {
        exitPresentationMode();
    }
}

function exitPresentationMode() {
    presentationMode = false;
    document.body.classList.remove('presentation-mode');
    document.querySelector('.navigation').style.display = 'flex';
    document.querySelector('.slide-counter').style.display = 'block';
}

// Slide jump functionality
function jumpToSlide(slideNumber) {
    if (slideNumber >= 1 && slideNumber <= totalSlides) {
        currentSlide = slideNumber;
        showSlide(currentSlide);
        logSlideTransition('jump', currentSlide);
    }
}

// URL hash handling for bookmarking
function handleHashChange() {
    const hash = window.location.hash;
    if (hash.startsWith('#slide-')) {
        const slideNumber = parseInt(hash.substring(7));
        if (slideNumber >= 1 && slideNumber <= totalSlides) {
            currentSlide = slideNumber;
            showSlide(currentSlide);
        }
    }
}

// Initialize from URL hash
function initializeFromHash() {
    handleHashChange();
}

// Event listeners for hash changes
window.addEventListener('hashchange', handleHashChange);

// Analytics and logging functions
function logSlideTransition(action, slideNumber) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] Slide transition: ${action} to slide ${slideNumber}`);
    
    // Could send to analytics service
    if (typeof gtag !== 'undefined') {
        gtag('event', 'slide_transition', {
            'slide_number': slideNumber,
            'action': action
        });
    }
}

function logInteraction(eventType, slideNumber, additionalData = {}) {
    const timestamp = new Date().toISOString();
    console.log(`[${timestamp}] Interaction: ${eventType} on slide ${slideNumber}`, additionalData);
    
    // Could send to analytics service
    if (typeof gtag !== 'undefined') {
        gtag('event', eventType, {
            'slide_number': slideNumber,
            ...additionalData
        });
    }
}

// Touch/swipe support for mobile devices
let touchStartX = 0;
let touchStartY = 0;
let touchEndX = 0;
let touchEndY = 0;

document.addEventListener('touchstart', function(e) {
    touchStartX = e.changedTouches[0].screenX;
    touchStartY = e.changedTouches[0].screenY;
});

document.addEventListener('touchend', function(e) {
    touchEndX = e.changedTouches[0].screenX;
    touchEndY = e.changedTouches[0].screenY;
    handleSwipe();
});

function handleSwipe() {
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;
    const minSwipeDistance = 50;
    
    // Check if horizontal swipe is more significant than vertical
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > minSwipeDistance) {
        if (deltaX > 0) {
            // Swipe right - previous slide
            previousSlide();
        } else {
            // Swipe left - next slide
            nextSlide();
        }
    }
}

// Auto-save progress
function saveProgress() {
    try {
        const progress = {
            currentSlide: currentSlide,
            timestamp: new Date().toISOString()
        };
        localStorage.setItem('cnn_presentation_progress', JSON.stringify(progress));
    } catch (error) {
        console.log('Could not save progress:', error);
    }
}

function loadProgress() {
    try {
        const saved = localStorage.getItem('cnn_presentation_progress');
        if (saved) {
            const progress = JSON.parse(saved);
            // Only restore if saved within last 24 hours
            const savedTime = new Date(progress.timestamp);
            const now = new Date();
            const hoursDiff = (now - savedTime) / (1000 * 60 * 60);
            
            if (hoursDiff < 24 && progress.currentSlide >= 1 && progress.currentSlide <= totalSlides) {
                currentSlide = progress.currentSlide;
                console.log(`Restored progress: slide ${currentSlide}`);
            }
        }
    } catch (error) {
        console.log('Could not load progress:', error);
    }
}

// Save progress whenever slide changes
function saveProgressOnSlideChange() {
    saveProgress();
}

// Performance monitoring
function trackPerformance() {
    if ('performance' in window) {
        window.addEventListener('load', function() {
            setTimeout(function() {
                const perfData = performance.getEntriesByType('navigation')[0];
                console.log('Page load performance:', {
                    loadTime: perfData.loadEventEnd - perfData.loadEventStart,
                    domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                    totalTime: perfData.loadEventEnd - perfData.fetchStart
                });
            }, 0);
        });
    }
}

// Accessibility enhancements
function enhanceAccessibility() {
    // Add ARIA labels to navigation buttons
    const prevBtn = document.querySelector('.nav-btn:first-child');
    const nextBtn = document.querySelector('.nav-btn:last-child');
    
    if (prevBtn) {
        prevBtn.setAttribute('aria-label', 'Diapositiva anterior');
        prevBtn.setAttribute('role', 'button');
    }
    
    if (nextBtn) {
        nextBtn.setAttribute('aria-label', 'Siguiente diapositiva');
        nextBtn.setAttribute('role', 'button');
    }
    
    // Add slide region ARIA label
    const slides = document.querySelectorAll('.slide');
    slides.forEach((slide, index) => {
        slide.setAttribute('role', 'region');
        slide.setAttribute('aria-label', `Diapositiva ${index + 1} de ${totalSlides}`);
    });
    
    // Add keyboard shortcuts info
    console.log('Atajos de teclado disponibles:');
    console.log('← / → : Navegar diapositivas');
    console.log('Espacio: Siguiente diapositiva');
    console.log('Home/End: Primera/Última diapositiva');
    console.log('Ctrl+F: Pantalla completa');
    console.log('Ctrl+P: Modo presentación');
    console.log('Escape: Salir de pantalla completa/modo presentación');
}

// Error handling and recovery
function handleError(error, context) {
    console.error(`Error in ${context}:`, error);
    
    // Attempt to recover from common errors
    if (context === 'slide_transition') {
        // If slide transition fails, try to reset to first slide
        try {
            currentSlide = 1;
            showSlide(1);
            console.log('Recovered from slide transition error');
        } catch (recoveryError) {
            console.error('Failed to recover from slide error:', recoveryError);
        }
    }
    
    // Log error for debugging
    logInteraction('error', currentSlide, {
        error: error.message,
        context: context
    });
}

// Slide timer for presentation mode
let slideTimer = null;
let slideTimeouts = {
    1: 30000,  // Title slide - 30 seconds
    2: 45000,  // Justificación - 45 seconds
    3: 60000,  // Descripción del problema - 1 minute
    // Add more timings as needed
};

function startSlideTimer(slideNumber) {
    if (slideTimer) {
        clearTimeout(slideTimer);
    }
    
    const timeout = slideTimeouts[slideNumber] || 60000; // Default 1 minute
    
    slideTimer = setTimeout(() => {
        if (presentationMode && currentSlide < totalSlides) {
            nextSlide();
        }
    }, timeout);
}

function stopSlideTimer() {
    if (slideTimer) {
        clearTimeout(slideTimer);
        slideTimer = null;
    }
}

// Enhanced slide content validation
function validateSlideContent() {
    const slides = document.querySelectorAll('.slide');
    let validationErrors = [];
    
    slides.forEach((slide, index) => {
        const slideNumber = index + 1;
        
        // Check for required elements
        const hasTitle = slide.querySelector('h1, h2');
        if (!hasTitle) {
            validationErrors.push(`Slide ${slideNumber}: Missing title`);
        }
        
        // Check for MathJax elements
        const mathElements = slide.querySelectorAll('[data-math], .math');
        mathElements.forEach(element => {
            if (!element.textContent.trim()) {
                validationErrors.push(`Slide ${slideNumber}: Empty math element`);
            }
        });
        
        // Check for broken images
        const images = slide.querySelectorAll('img');
        images.forEach(img => {
            img.onerror = function() {
                validationErrors.push(`Slide ${slideNumber}: Broken image - ${this.src}`);
            };
        });
        
        // Check for audio elements
        const audios = slide.querySelectorAll('audio');
        audios.forEach(audio => {
            audio.onerror = function() {
                validationErrors.push(`Slide ${slideNumber}: Audio loading error - ${this.src}`);
            };
        });
    });
    
    if (validationErrors.length > 0) {
        console.warn('Slide validation errors:', validationErrors);
    } else {
        console.log('All slides validated successfully');
    }
    
    return validationErrors;
}

// Export presentation data
function exportPresentationData() {
    const presentationData = {
        title: 'Análisis de Emociones en la Voz con CNN 1D',
        totalSlides: totalSlides,
        currentSlide: currentSlide,
        timestamp: new Date().toISOString(),
        slides: []
    };
    
    const slides = document.querySelectorAll('.slide');
    slides.forEach((slide, index) => {
        const slideData = {
            number: index + 1,
            title: slide.querySelector('h1, h2')?.textContent || 'Sin título',
            hasFormulas: slide.querySelectorAll('.formula-block').length > 0,
            hasImages: slide.querySelectorAll('img').length > 0,
            hasAudio: slide.querySelectorAll('audio').length > 0,
            hasTables: slide.querySelectorAll('table').length > 0,
            hasInteractiveElements: slide.querySelectorAll('.layer-box, .cycle-step').length > 0
        };
        presentationData.slides.push(slideData);
    });
    
    return presentationData;
}

// Print presentation summary
function printPresentationSummary() {
    const data = exportPresentationData();
    console.log('=== RESUMEN DE LA PRESENTACIÓN ===');
    console.log(`Título: ${data.title}`);
    console.log(`Total de diapositivas: ${data.totalSlides}`);
    console.log(`Diapositiva actual: ${data.currentSlide}`);
    console.log('\n=== CONTENIDO POR DIAPOSITIVA ===');
    
    data.slides.forEach(slide => {
        console.log(`\nDiapositiva ${slide.number}: ${slide.title}`);
        const features = [];
        if (slide.hasFormulas) features.push('Fórmulas matemáticas');
        if (slide.hasImages) features.push('Imágenes');
        if (slide.hasAudio) features.push('Audio');
        if (slide.hasTables) features.push('Tablas');
        if (slide.hasInteractiveElements) features.push('Elementos interactivos');
        
        if (features.length > 0) {
            console.log(`  Contiene: ${features.join(', ')}`);
        }
    });
}

// Development helpers
function enableDevelopmentMode() {
    if (location.hostname === 'localhost' || location.hostname === '127.0.0.1') {
        // Add development shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.shiftKey) {
                switch(e.key) {
                    case 'D':
                        e.preventDefault();
                        console.log('=== DEBUG INFO ===');
                        console.log('Current slide:', currentSlide);
                        console.log('Total slides:', totalSlides);
                        console.log('Presentation mode:', presentationMode);
                        printPresentationSummary();
                        break;
                    case 'V':
                        e.preventDefault();
                        validateSlideContent();
                        break;
                    case 'R':
                        e.preventDefault();
                        location.reload();
                        break;
                }
            }
        });
        
        console.log('Development mode enabled');
        console.log('Debug shortcuts: Ctrl+Shift+D (debug), Ctrl+Shift+V (validate), Ctrl+Shift+R (reload)');
    }
}

// Initialize presentation
function initializePresentation() {
    try {
        console.log('Inicializando presentación técnica CNN 1D...');
        
        // Load saved progress
        loadProgress();
        
        // Initialize from URL hash
        initializeFromHash();
        
        // Show initial slide
        showSlide(currentSlide);
        
        // Enhance accessibility
        enhanceAccessibility();
        
        // Validate slide content
        setTimeout(validateSlideContent, 1000);
        
        // Track performance
        trackPerformance();
        
        // Enable development mode in local environment
        enableDevelopmentMode();
        
        // Add event listeners for progress saving
        ['beforeunload', 'pagehide'].forEach(event => {
            window.addEventListener(event, saveProgressOnSlideChange);
        });
        
        // Add visibility change handler
        document.addEventListener('visibilitychange', function() {
            if (document.visibilityState === 'hidden') {
                saveProgress();
                // Pause any playing audio
                document.querySelectorAll('audio').forEach(audio => audio.pause());
            }
        });
        
        console.log('Presentación inicializada correctamente');
        console.log(`Diapositiva actual: ${currentSlide}/${totalSlides}`);
        
        // Print keyboard shortcuts
        setTimeout(() => {
            console.log('\n=== ATAJOS DE TECLADO ===');
            console.log('Navegación:');
            console.log('  ← / → : Diapositiva anterior/siguiente');
            console.log('  Espacio: Siguiente diapositiva');
            console.log('  Home/End: Primera/última diapositiva');
            console.log('\nModos:');
            console.log('  Ctrl+F: Pantalla completa');
            console.log('  Ctrl+P: Modo presentación');
            console.log('  Escape: Salir de modos especiales');
            console.log('\nTáctil:');
            console.log('  Deslizar izq/der: Cambiar diapositiva');
        }, 2000);
        
    } catch (error) {
        handleError(error, 'initialization');
    }
}

// Override the original showSlide to add enhanced features
const originalShowSlide = showSlide;
showSlide = function(n) {
    try {
        originalShowSlide(n);
        
        // Start slide timer if in presentation mode
        if (presentationMode) {
            startSlideTimer(n);
        }
        
        // Save progress
        saveProgressOnSlideChange();
        
    } catch (error) {
        handleError(error, 'slide_transition');
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePresentation);
} else {
    initializePresentation();
}

// Export functions for external use
window.PresentationAPI = {
    jumpToSlide,
    toggleFullscreen,
    togglePresentationMode,
    exportPresentationData,
    printPresentationSummary,
    getCurrentSlide: () => currentSlide,
    getTotalSlides: () => totalSlides,
    isPresentationMode: () => presentationMode
};