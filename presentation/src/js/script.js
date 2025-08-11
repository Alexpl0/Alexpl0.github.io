// ============================================================================
// SIMPLIFIED CNN 1D PRESENTATION - JAVASCRIPT
// Sistema de presentación técnica - Versión simplificada
// ============================================================================

// ============================================================================
// CONFIGURACIÓN Y ESTADO GLOBAL
// ============================================================================

let currentSlide = 1;
const totalSlides = 57; // Total de diapositivas

// ============================================================================
// NAVEGACIÓN PRINCIPAL
// ============================================================================

function showSlide(n) {
    if (n < 1 || n > totalSlides) {
        return false;
    }

    // Ocultar todas las diapositivas
    document.querySelectorAll('.slide').forEach(slide => {
        slide.style.display = 'none';
        slide.classList.add('hidden');
    });
    
    // Mostrar diapositiva actual
    const current = document.querySelector(`[data-slide="${n}"]`);
    if (current) {
        current.style.display = 'flex';
        current.classList.remove('hidden');
        
        // Aplicar transición suave
        current.style.opacity = '0';
        current.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            current.style.transition = 'all 0.6s ease-out';
            current.style.opacity = '1';
            current.style.transform = 'translateY(0)';
        }, 50);
    }
    
    updateUI(n);
    return true;
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        showSlide(currentSlide);
    } else {
        // Efecto visual al final
        showMessage('🎉 ¡Has completado la presentación! 🎉', 3000);
    }
}

function previousSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        showSlide(currentSlide);
    } else {
        showMessage('📍 Ya estás en la primera diapositiva', 2000);
    }
}

function jumpToSlide(slideNumber) {
    if (slideNumber >= 1 && slideNumber <= totalSlides) {
        currentSlide = slideNumber;
        showSlide(currentSlide);
        return true;
    }
    return false;
}

// ============================================================================
// ACTUALIZACIÓN DE INTERFAZ
// ============================================================================

function updateUI(n) {
    updateSlideCounter(n);
    updateProgressBar(n);
    updateURL(n);
}

function updateSlideCounter(n) {
    const counter = document.getElementById('slideCounter');
    if (counter) {
        counter.textContent = `${n} / ${totalSlides}`;
        // Pequeña animación en el contador
        counter.style.transform = 'scale(1.1)';
        setTimeout(() => counter.style.transform = 'scale(1)', 200);
    }
}

function updateProgressBar(n) {
    const progress = (n / totalSlides) * 100;
    const progressFill = document.getElementById('progressFill');
    if (progressFill) {
        progressFill.style.width = `${progress}%`;
    }
}

function updateURL(n) {
    if (history.replaceState) {
        history.replaceState(null, null, `#slide-${n}`);
    }
}

// ============================================================================
// NAVEGACIÓN POR TECLADO
// ============================================================================

function handleKeyboardNavigation(e) {
    // Ignorar si hay elementos focusados (inputs, etc.)
    if (isInputFocused()) return;
    
    const keyActions = {
        'ArrowRight': () => nextSlide(),
        'ArrowDown': () => nextSlide(),
        ' ': () => nextSlide(),
        'PageDown': () => nextSlide(),
        'ArrowLeft': () => previousSlide(),
        'ArrowUp': () => previousSlide(),
        'PageUp': () => previousSlide(),
        'Home': () => jumpToSlide(1),
        'End': () => jumpToSlide(totalSlides)
    };
    
    // Navegación directa con números 1-9
    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const num = parseInt(e.key);
        if (num >= 1 && num <= 9) {
            e.preventDefault();
            jumpToSlide(num);
            return;
        }
    }
    
    if (keyActions[e.key]) {
        e.preventDefault();
        keyActions[e.key]();
    }
}

function isInputFocused() {
    const activeElement = document.activeElement;
    return activeElement.tagName === 'INPUT' || 
           activeElement.tagName === 'TEXTAREA' ||
           activeElement.contentEditable === 'true' ||
           activeElement.tagName === 'SELECT';
}

// ============================================================================
// GESTIÓN DE URL HASH
// ============================================================================

function handleHashChange() {
    const hash = window.location.hash;
    if (hash.startsWith('#slide-')) {
        const slideNumber = parseInt(hash.substring(7));
        if (slideNumber >= 1 && slideNumber <= totalSlides && slideNumber !== currentSlide) {
            currentSlide = slideNumber;
            showSlide(currentSlide);
        }
    }
}

// ============================================================================
// MENSAJES Y NOTIFICACIONES
// ============================================================================

function showMessage(message, duration = 3000) {
    const messageDiv = document.createElement('div');
    messageDiv.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #0726D9, #3D9DF2); color: white;
        padding: 20px 40px; border-radius: 15px; font-size: 1.2rem; font-weight: bold;
        z-index: 10000; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: message-appear 0.5s ease-out;
        pointer-events: none;
    `;
    messageDiv.textContent = message;
    
    // Añadir estilos de animación si no existen
    if (!document.querySelector('#message-style')) {
        const style = document.createElement('style');
        style.id = 'message-style';
        style.textContent = `
            @keyframes message-appear {
                from { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
            }
            @keyframes message-disappear {
                from { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                to { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.style.animation = 'message-disappear 0.5s ease-out forwards';
        setTimeout(() => messageDiv.remove(), 500);
    }, duration);
}

// ============================================================================
// INICIALIZACIÓN Y CONFIGURACIÓN DE EVENTOS
// ============================================================================

function initializePresentation() {
    console.log('🚀 Inicializando presentación CNN 1D...');
    
    try {
        // Manejar URL hash inicial
        handleHashChange();
        
        // Mostrar diapositiva inicial si no hay hash
        if (!window.location.hash) {
            showSlide(currentSlide);
        }
        
        // Configurar event listeners
        setupEventListeners();
        
        // Configurar MathJax si está disponible
        if (typeof MathJax !== 'undefined') {
            MathJax.config.tex.inlineMath = [['$', '$'], ['\\(', '\\)']];
            MathJax.config.tex.displayMath = [['$$', '$$'], ['\\[', '\\]']];
        }
        
        console.log('✅ Presentación inicializada correctamente');
        console.log(`📊 Diapositiva: ${currentSlide}/${totalSlides}`);
        
    } catch (error) {
        console.error('❌ Error inicializando presentación:', error);
        showMessage('Error al inicializar la presentación.', 4000);
    }
}

function setupEventListeners() {
    // Navegación por teclado
    document.addEventListener('keydown', handleKeyboardNavigation);
    
    // Cambios de URL hash
    window.addEventListener('hashchange', handleHashChange);
    
    // Prevenir scroll con teclas de navegación
    document.addEventListener('keydown', function(e) {
        if (['ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End', ' '].includes(e.key)) {
            if (!isInputFocused()) {
                e.preventDefault();
            }
        }
    });
    
    console.log('🎛️ Event listeners configurados');
}

// ============================================================================
// UTILIDADES ADICIONALES
// ============================================================================

function preloadImages() {
    // Precargar imágenes de las siguientes diapositivas para mejor rendimiento
    const nextSlideNumbers = [currentSlide + 1, currentSlide + 2].filter(n => n <= totalSlides);
    
    nextSlideNumbers.forEach(slideNum => {
        const slide = document.querySelector(`[data-slide="${slideNum}"]`);
        if (slide) {
            const images = slide.querySelectorAll('img');
            images.forEach(img => {
                if (img.src && !img.complete) {
                    const preloadImg = new Image();
                    preloadImg.src = img.src;
                }
            });
        }
    });
}

// Función para obtener el estado actual
function getCurrentState() {
    return {
        currentSlide,
        totalSlides,
        progress: (currentSlide / totalSlides) * 100
    };
}

// ============================================================================
// EXPOSICIÓN GLOBAL Y AUTO-INICIALIZACIÓN
// ============================================================================

// Exponer funciones principales al scope global para compatibilidad con HTML
window.nextSlide = nextSlide;
window.previousSlide = previousSlide;
window.jumpToSlide = jumpToSlide;
window.showSlide = showSlide;
window.getCurrentState = getCurrentState;

// Auto-inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePresentation);
} else {
    initializePresentation();
}

// Precargar imágenes cuando se cambie de diapositiva
document.addEventListener('keydown', () => {
    setTimeout(preloadImages, 100);
});

// Mensaje de consola informativo
console.log(`
🎯 CNN 1D Presentation - Versión Simplificada
============================================
📊 Total Slides: ${totalSlides}
⌨️  Keyboard shortcuts: HABILITADO
🎛️ Navegación básica: LISTA

🚀 CONTROLES:
• ← → ↑ ↓ : Navegación
• Home/End : Primera/Última
• 1-9 : Ir a diapositiva específica
• Espacio/PageDown : Siguiente
• PageUp : Anterior
`);