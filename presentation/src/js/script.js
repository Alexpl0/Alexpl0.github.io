// ============================================================================
// ENHANCED CNN 1D PRESENTATION - JAVASCRIPT COMPLETO
// Sistema de presentación técnica interactiva
// Versión completa y optimizada - CORREGIDA
// ============================================================================

// ============================================================================
// CONFIGURACIÓN Y ESTADO GLOBAL
// ============================================================================

// Variables principales
let currentSlide = 1;
const totalSlides = 48; // Ajustado al número real de diapositivas

// Estado de la aplicación
const presentationState = {
    mode: 'normal',
    autoAdvance: false,
    autoAdvanceInterval: null,
    slideHistory: [],
    bookmarks: new Set(),
    analytics: {
        startTime: Date.now(),
        slideViews: new Map(),
        interactions: [],
        lastSlide: null,
        lastSlideTime: null
    }
};

// Configuración del sistema
const config = {
    autoAdvanceDelay: 30000,
    keyboardShortcuts: true,
    touchGestures: true,
    analytics: true,
    accessibility: true,
    animations: true,
    preloadImages: true
};

// Variables para gestos táctiles
let touchStartX = 0;
let touchStartY = 0;
let touchEndX = 0;
let touchEndY = 0;
let isTouching = false;

// ============================================================================
// NAVEGACIÓN PRINCIPAL
// ============================================================================

function showSlide(n, source = 'navigation') {
    if (n < 1 || n > totalSlides) {
        console.warn(`Slide ${n} out of range`);
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
        
        if (config.animations) {
            triggerSlideAnimation(current);
        }
        
        updateSlideFeatures(n);
        updateSlideHistory(n);
        
        if (config.analytics) {
            trackSlideView(n, source);
        }
    }
    
    updateUI(n);
    preloadNearbySlides(n);
    
    return true;
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        showSlide(currentSlide, 'next');
        logInteraction('slide_next', currentSlide);
    } else {
        showEndEffect();
    }
}

function previousSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        showSlide(currentSlide, 'previous');
        logInteraction('slide_previous', currentSlide);
    } else {
        showStartEffect();
    }
}

function jumpToSlide(slideNumber, source = 'jump') {
    if (slideNumber >= 1 && slideNumber <= totalSlides) {
        currentSlide = slideNumber;
        showSlide(currentSlide, source);
        logInteraction('slide_jump', currentSlide, { to: slideNumber });
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
        counter.style.transform = 'scale(1.1)';
        setTimeout(() => counter.style.transform = 'scale(1)', 200);
    }
}

function updateProgressBar(n) {
    const progress = (n / totalSlides) * 100;
    const progressFill = document.getElementById('progressFill');
    if (progressFill) {
        progressFill.style.width = `${progress}%`;
        progressFill.style.boxShadow = '0 0 20px rgba(61, 157, 242, 0.8)';
        setTimeout(() => progressFill.style.boxShadow = '0 0 10px rgba(61, 157, 242, 0.5)', 300);
    }
}

function updateURL(n) {
    if (history.replaceState) {
        history.replaceState(null, null, `#slide-${n}`);
    }
}

function updateSlideHistory(n) {
    if (presentationState.slideHistory[presentationState.slideHistory.length - 1] !== n) {
        presentationState.slideHistory.push(n);
    }
}

// ============================================================================
// CARACTERÍSTICAS ESPECÍFICAS DE DIAPOSITIVAS
// ============================================================================

function updateSlideFeatures(slideNumber) {
    // MathJax
    if (typeof MathJax !== 'undefined' && MathJax.typesetPromise) {
        MathJax.typesetPromise().catch(err => console.log('MathJax error:', err));
    }
    
    initializeSlideAnimations(slideNumber);
    updateAudioControls(slideNumber);
    initializeInteractiveElements(slideNumber);
    configureSlideSpecifics(slideNumber);
}

function configureSlideSpecifics(slideNumber) {
    const configurations = {
        6: initializeCycleDiagram,
        21: initializeInteractivePlots,
        22: initializeInteractivePlots,
        35: initializeArchitectureSummary,
        40: initializeConfusionMatrix
    };
    
    if (configurations[slideNumber]) {
        configurations[slideNumber]();
    }
}

// ============================================================================
// ANIMACIONES Y EFECTOS VISUALES
// ============================================================================

function triggerSlideAnimation(slideElement) {
    slideElement.style.opacity = '0';
    slideElement.style.transform = 'translateY(30px)';
    
    requestAnimationFrame(() => {
        slideElement.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)';
        slideElement.style.opacity = '1';
        slideElement.style.transform = 'translateY(0)';
    });
}

function initializeSlideAnimations(slideNumber) {
    const currentSlide = document.querySelector(`[data-slide="${slideNumber}"]`);
    if (!currentSlide) return;
    
    // Animar elementos técnicos
    animateElements(currentSlide.querySelectorAll('.technical-step-layout, .neural-architecture, .formula-block, .layer-box'), 150);
    
    // Animar tablas
    animateTables(currentSlide.querySelectorAll('.comparison-table, .mfcc-table, .cnn-table'));
    
    // Animar estadísticas
    animateStats(currentSlide.querySelectorAll('.stat-item'));
    
    // Animar gráficos
    animateCharts(currentSlide.querySelectorAll('.single-chart-container img'));
}

function animateElements(elements, delay = 150) {
    elements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            element.style.transition = 'all 0.6s ease-out';
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }, 100 + index * delay);
    });
}

function animateTables(tables) {
    tables.forEach((table, tableIndex) => {
        const rows = table.querySelectorAll('tr');
        rows.forEach((row, rowIndex) => {
            row.style.opacity = '0';
            row.style.transform = 'translateX(-20px)';
            
            setTimeout(() => {
                row.style.transition = 'all 0.4s ease-out';
                row.style.opacity = '1';
                row.style.transform = 'translateX(0)';
            }, 200 + tableIndex * 100 + rowIndex * 50);
        });
    });
}

function animateStats(statItems) {
    statItems.forEach((item, index) => {
        item.style.opacity = '0';
        item.style.transform = 'scale(0.8)';
        
        setTimeout(() => {
            item.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
            item.style.opacity = '1';
            item.style.transform = 'scale(1)';
            
            // Contador animado para números
            animateCounter(item);
        }, 300 + index * 100);
    });
}

function animateCharts(charts) {
    charts.forEach((chart, index) => {
        chart.style.opacity = '0';
        chart.style.transform = 'scale(0.9)';
        
        setTimeout(() => {
            chart.style.transition = 'all 0.6s ease-out';
            chart.style.opacity = '1';
            chart.style.transform = 'scale(1)';
        }, 400 + index * 200);
    });
}

function animateCounter(statItem) {
    const numberElement = statItem.querySelector('h3');
    if (!numberElement) return;
    
    const text = numberElement.textContent.trim();
    const match = text.match(/^(\d+(?:\.\d+)?)/);
    
    if (match) {
        const targetNumber = parseFloat(match[1]);
        const suffix = text.replace(match[0], '');
        let currentNumber = 0;
        const increment = targetNumber / 30; // 30 frames para la animación
        
        const countUp = () => {
            currentNumber += increment;
            if (currentNumber >= targetNumber) {
                numberElement.textContent = targetNumber + suffix;
            } else {
                numberElement.textContent = Math.floor(currentNumber) + suffix;
                requestAnimationFrame(countUp);
            }
        };
        
        countUp();
    }
}

function showEndEffect() {
    createConfetti();
    showMessage('🎉 ¡Has completado la presentación! 🎉', 3000);
    logInteraction('presentation_completed', totalSlides);
}

function showStartEffect() {
    showMessage('📍 Ya estás en la primera diapositiva', 2000);
}

function createConfetti() {
    const colors = ['#0726D9', '#1760BF', '#3D9DF2', '#52C5F2'];
    
    for (let i = 0; i < 50; i++) {
        const confetti = document.createElement('div');
        confetti.style.cssText = `
            position: fixed; width: 10px; height: 10px; border-radius: 50%;
            background: ${colors[Math.floor(Math.random() * colors.length)]};
            top: -10px; left: ${Math.random() * 100}vw; z-index: 9999;
            pointer-events: none; animation: confetti-fall 3s ease-out forwards;
        `;
        
        document.body.appendChild(confetti);
        setTimeout(() => confetti.remove(), 3000);
    }
    
    addConfettiCSS();
}

function addConfettiCSS() {
    if (document.querySelector('#confetti-style')) return;
    
    const style = document.createElement('style');
    style.id = 'confetti-style';
    style.textContent = `
        @keyframes confetti-fall {
            to { transform: translateY(100vh) rotate(360deg); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
}

function showMessage(message, duration = 3000) {
    const messageDiv = document.createElement('div');
    messageDiv.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #0726D9, #3D9DF2); color: white;
        padding: 20px 40px; border-radius: 15px; font-size: 1.2rem; font-weight: bold;
        z-index: 10000; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        animation: message-appear 0.5s ease-out;
    `;
    messageDiv.textContent = message;
    
    addMessageCSS();
    document.body.appendChild(messageDiv);
    
    setTimeout(() => {
        messageDiv.style.animation = 'message-disappear 0.5s ease-out forwards';
        setTimeout(() => messageDiv.remove(), 500);
    }, duration);
}

function addMessageCSS() {
    if (document.querySelector('#message-style')) return;
    
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

// ============================================================================
// ELEMENTOS INTERACTIVOS
// ============================================================================

function initializeInteractiveElements(slideNumber) {
    const currentSlide = document.querySelector(`[data-slide="${slideNumber}"]`);
    if (!currentSlide) return;
    
    initializeInteractiveTables(currentSlide);
    initializeFormulaBlocks(currentSlide);
    initializeLayerBoxes(currentSlide);
    initializeCodeBlocks(currentSlide);
    initializeHoverEffects(currentSlide);
}

function initializeInteractiveTables(slideElement) {
    const tables = slideElement.querySelectorAll('.comparison-table, .mfcc-table, .cnn-table');
    
    tables.forEach(table => {
        const rows = table.querySelectorAll('tbody tr');
        rows.forEach((row, index) => {
            addTableRowInteractivity(row, table, index);
        });
    });
}

function addTableRowInteractivity(row, table, index) {
    row.addEventListener('mouseenter', () => {
        row.style.backgroundColor = 'rgba(61, 157, 242, 0.15)';
        row.style.transform = 'translateX(3px)';
        row.style.transition = 'all 0.2s ease';
        row.style.cursor = 'pointer';
    });
    
    row.addEventListener('mouseleave', () => {
        row.style.backgroundColor = '';
        row.style.transform = 'translateX(0)';
    });
    
    row.addEventListener('click', () => {
        highlightSelectedRow(row, table);
        showRowDetails(table, index);
        logInteraction('table_row_click', currentSlide, { table: table.className, row: index });
    });
}

function highlightSelectedRow(selectedRow, table) {
    table.querySelectorAll('tbody tr').forEach(r => r.classList.remove('selected'));
    selectedRow.classList.add('selected');
    
    // Añadir estilo visual para fila seleccionada
    if (!document.querySelector('#selected-row-style')) {
        const style = document.createElement('style');
        style.id = 'selected-row-style';
        style.textContent = `
            .selected {
                background: linear-gradient(90deg, rgba(7, 38, 217, 0.1), rgba(61, 157, 242, 0.1)) !important;
                border-left: 4px solid #0726D9 !important;
                font-weight: 600 !important;
            }
        `;
        document.head.appendChild(style);
    }
}

function showRowDetails(table, rowIndex) {
    const tableType = table.className;
    
    if (tableType.includes('comparison-table')) {
        showComparisonDetails(table, rowIndex);
    } else if (tableType.includes('cnn-table')) {
        showCNNLayerDetails(rowIndex);
    } else if (tableType.includes('mfcc-table')) {
        showMFCCDetails(rowIndex);
    }
}

function showComparisonDetails(table, rowIndex) {
    const rows = table.querySelectorAll('tbody tr');
    const selectedRow = rows[rowIndex];
    
    if (selectedRow) {
        const cells = selectedRow.querySelectorAll('td');
        const details = Array.from(cells).map(cell => cell.textContent).join(' | ');
        showMessage(`Detalles: ${details}`, 3000);
    }
}

function showCNNLayerDetails(rowIndex) {
    const layerDetails = {
        0: 'Entrada: Recibe vector de 180 características extraídas del audio usando Librosa',
        1: 'Conv1D_1: Primera capa convolucional con 128 filtros de tamaño 5 para detectar patrones locales',
        2: 'BatchNorm_1: Normaliza activaciones para estabilizar entrenamiento y acelerar convergencia',
        3: 'MaxPool_1: Reduce dimensionalidad de 180 a 90 tomando el máximo de cada ventana',
        4: 'Dropout_1: Desactiva 30% de neuronas aleatoriamente para prevenir overfitting',
        5: 'Conv1D_2: Segunda capa convolucional con 64 filtros para detectar patrones complejos',
        6: 'BatchNorm_2: Segunda normalización por lotes para estabilidad',
        7: 'MaxPool_2: Segunda reducción de dimensionalidad de 90 a 45',
        8: 'Dropout_2: Segunda capa de dropout para regularización adicional',
        9: 'GlobalAvgPool: Promedia cada canal en toda la secuencia temporal',
        10: 'Dense_Inter: Capa densa intermedia con 64 neuronas y activación ReLU',
        11: 'Dropout_Final: Último dropout antes de la clasificación',
        12: 'Output: Capa final con 7 neuronas y activación Softmax para clasificación'
    };
    
    const detail = layerDetails[rowIndex] || 'Información de capa no disponible';
    showMessage(detail, 4000);
}

function showMFCCDetails(rowIndex) {
    const mfccDetails = {
        0: 'MFCC 0: Energía total o sonoridad de la señal - representa la intensidad general del audio',
        1: 'MFCC 1-4: Capturan la forma general y pendiente del espectro - contornos principales de la voz',
        2: 'MFCC 5-13+: Describen detalles finos y texturas del espectro - características específicas del timbre'
    };
    
    const detail = mfccDetails[rowIndex] || 'Información de MFCC no disponible';
    showMessage(detail, 4000);
}

function initializeFormulaBlocks(slideElement) {
    const formulaBlocks = slideElement.querySelectorAll('.formula-block');
    
    formulaBlocks.forEach((block, index) => {
        block.style.cursor = 'pointer';
        block.title = 'Click para expandir/contraer explicación';
        
        block.addEventListener('click', () => {
            toggleFormulaBlock(block, index);
        });
        
        block.addEventListener('mouseenter', () => {
            block.style.transform = 'translateY(-2px)';
            block.style.boxShadow = '0 10px 25px rgba(3, 18, 64, 0.15)';
        });
        
        block.addEventListener('mouseleave', () => {
            if (!block.classList.contains('expanded')) {
                block.style.transform = 'translateY(0)';
                block.style.boxShadow = '';
            }
        });
    });
}

function toggleFormulaBlock(block, index) {
    block.classList.toggle('expanded');
    
    if (block.classList.contains('expanded')) {
        expandFormulaBlock(block);
    } else {
        contractFormulaBlock(block);
    }
    
    logInteraction('formula_block_toggle', currentSlide, { 
        expanded: block.classList.contains('expanded'),
        formula: index 
    });
}

function expandFormulaBlock(block) {
    block.style.transform = 'scale(1.05)';
    block.style.boxShadow = '0 15px 35px rgba(3, 18, 64, 0.25)';
    block.style.zIndex = '10';
    block.style.position = 'relative';
    
    showFormulaExplanation(block);
}

function contractFormulaBlock(block) {
    block.style.transform = 'scale(1)';
    block.style.boxShadow = '';
    block.style.zIndex = '';
    block.style.position = '';
    
    hideFormulaExplanation(block);
}

function showFormulaExplanation(formulaBlock) {
    const explanationDiv = document.createElement('div');
    explanationDiv.className = 'formula-explanation';
    explanationDiv.style.cssText = `
        position: absolute; top: 100%; left: 0; right: 0;
        background: white; border: 2px solid #3D9DF2; border-radius: 10px;
        padding: 15px; margin-top: 10px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        z-index: 11; font-size: 0.9rem; line-height: 1.4;
        animation: explanation-appear 0.3s ease-out;
    `;
    
    const explanation = getFormulaExplanation(formulaBlock.textContent);
    explanationDiv.innerHTML = `
        <div style="color: #0726D9; font-weight: bold; margin-bottom: 8px;">💡 Explicación:</div>
        <div style="color: #333;">${explanation}</div>
    `;
    
    formulaBlock.appendChild(explanationDiv);
    
    // Añadir animación
    if (!document.querySelector('#explanation-style')) {
        const style = document.createElement('style');
        style.id = 'explanation-style';
        style.textContent = `
            @keyframes explanation-appear {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
        `;
        document.head.appendChild(style);
    }
}

function getFormulaExplanation(formulaText) {
    if (formulaText.includes('ReLU')) {
        return 'ReLU (Rectified Linear Unit) es una función de activación que devuelve 0 para valores negativos y el valor original para valores positivos. Introduce no-linealidad sin el problema del gradiente desvaneciente.';
    } else if (formulaText.includes('Softmax')) {
        return 'Softmax convierte un vector de puntuaciones en una distribución de probabilidad. Cada elemento resultante está entre 0 y 1, y la suma total es 1.';
    } else if (formulaText.includes('Adam') || formulaText.includes('ADAM')) {
        return 'ADAM es un optimizador adaptativo que combina las ventajas de AdaGrad y RMSprop, ajustando la tasa de aprendizaje individualmente para cada parámetro.';
    } else if (formulaText.includes('StandardScaler') || formulaText.includes('z-score')) {
        return 'Z-score normaliza los datos restando la media y dividiendo por la desviación estándar, centrando los datos en 0 con varianza 1.';
    } else if (formulaText.includes('FFT') || formulaText.includes('Fourier')) {
        return 'La Transformada de Fourier convierte señales del dominio temporal al dominio de frecuencia, revelando qué frecuencias componen la señal.';
    } else if (formulaText.includes('MFCC')) {
        return 'Los MFCCs imitan la percepción auditiva humana aplicando filtros mel, logaritmo y DCT para extraer características compactas del espectro.';
    }
    return 'Esta fórmula representa un concepto matemático fundamental en el modelo de reconocimiento de emociones.';
}

function hideFormulaExplanation(formulaBlock) {
    const explanation = formulaBlock.querySelector('.formula-explanation');
    if (explanation) {
        explanation.style.animation = 'explanation-disappear 0.3s ease-out forwards';
        setTimeout(() => explanation.remove(), 300);
    }
}

function initializeLayerBoxes(slideElement) {
    const layerBoxes = slideElement.querySelectorAll('.layer-box');
    
    layerBoxes.forEach((box, index) => {
        box.style.cursor = 'pointer';
        box.title = 'Click para ver detalles técnicos de la capa';
        
        addLayerBoxEvents(box, index);
    });
}

function addLayerBoxEvents(box, index) {
    box.addEventListener('click', () => {
        toggleLayerDetails(box, index);
    });
    
    box.addEventListener('mouseenter', () => {
        box.style.transform = 'translateY(-5px) scale(1.02)';
        box.style.boxShadow = '0 20px 40px rgba(3, 18, 64, 0.3)';
    });
    
    box.addEventListener('mouseleave', () => {
        if (!box.classList.contains('detailed')) {
            box.style.transform = 'translateY(0) scale(1)';
            box.style.boxShadow = '';
        }
    });
}

function toggleLayerDetails(box, index) {
    box.classList.toggle('detailed');
    
    if (box.classList.contains('detailed')) {
        showLayerDetails(index);
    } else {
        hideLayerDetails(box);
    }
    
    logInteraction('layer_box_click', currentSlide, { 
        layer: index,
        detailed: box.classList.contains('detailed')
    });
}

function showLayerDetails(index) {
    const layerInfo = [
        'Capa de entrada: Recibe las 180 características extraídas. No tiene parámetros entrenables.',
        'Conv1D_1: 128 filtros de tamaño 5 detectan patrones locales. Total: 768 parámetros.',
        'BatchNormalization: Normaliza las activaciones para acelerar convergencia. 512 parámetros.',
        'MaxPooling: Reduce dimensionalidad tomando el máximo de cada ventana. Sin parámetros.',
        'Dropout: Desactiva aleatoriamente 30% de neuronas durante entrenamiento.',
        'Conv1D_2: 64 filtros procesan los 128 mapas anteriores. Total: 41,024 parámetros.',
        'GlobalAvgPooling: Promedia cada canal en toda la secuencia temporal.',
        'Dense: Capa final con 7 neuronas para clasificación. Total: 455 parámetros.'
    ];
    
    const info = layerInfo[index] || 'Información de capa no disponible.';
    showMessage(info, 5000);
}

function hideLayerDetails(box) {
    box.style.transform = 'translateY(0) scale(1)';
    box.style.boxShadow = '';
}

function initializeCodeBlocks(slideElement) {
    const codeBlocks = slideElement.querySelectorAll('pre code, pre');
    
    codeBlocks.forEach((block, index) => {
        addCopyButton(block);
        enhanceSyntaxHighlighting(block);
        
        block.addEventListener('click', () => {
            selectText(block);
            logInteraction('code_block_select', currentSlide, { block: index });
        });
    });
}

function addCopyButton(codeBlock) {
    const container = codeBlock.closest('.code-block-container') || codeBlock.parentElement;
    
    // Evitar duplicar botones
    if (container.querySelector('.copy-button')) return;
    
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-button';
    copyBtn.innerHTML = '📋 Copiar';
    copyBtn.style.cssText = `
        position: absolute; top: 10px; right: 10px;
        background: #0078D4; color: white; border: none;
        padding: 5px 10px; border-radius: 5px; cursor: pointer;
        font-size: 0.8rem; z-index: 1; transition: all 0.2s ease;
    `;
    
    copyBtn.addEventListener('mouseenter', () => {
        copyBtn.style.background = '#106ebe';
        copyBtn.style.transform = 'scale(1.05)';
    });
    
    copyBtn.addEventListener('mouseleave', () => {
        copyBtn.style.background = '#0078D4';
        copyBtn.style.transform = 'scale(1)';
    });
    
    copyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        const textToCopy = codeBlock.textContent || codeBlock.innerText;
        
        navigator.clipboard.writeText(textToCopy).then(() => {
            copyBtn.innerHTML = '✅ Copiado';
            copyBtn.style.background = '#28a745';
            setTimeout(() => {
                copyBtn.innerHTML = '📋 Copiar';
                copyBtn.style.background = '#0078D4';
            }, 2000);
            logInteraction('code_copy', currentSlide);
        }).catch(() => {
            // Fallback para navegadores sin clipboard API
            fallbackCopy(textToCopy, copyBtn);
        });
    });
    
    container.style.position = 'relative';
    container.appendChild(copyBtn);
}

function fallbackCopy(text, button) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    
    try {
        document.execCommand('copy');
        button.innerHTML = '✅ Copiado';
        button.style.background = '#28a745';
        setTimeout(() => {
            button.innerHTML = '📋 Copiar';
            button.style.background = '#0078D4';
        }, 2000);
    } catch (err) {
        button.innerHTML = '❌ Error';
        setTimeout(() => {
            button.innerHTML = '📋 Copiar';
        }, 2000);
    } finally {
        document.body.removeChild(textArea);
    }
}

function enhanceSyntaxHighlighting(codeBlock) {
    const code = codeBlock.textContent || codeBlock.innerText;
    
    // Solo aplicar si no tiene highlighting previo
    if (codeBlock.querySelector('.highlight')) return;
    
    const highlighted = code
        .replace(/(from|import|def|class|if|else|elif|for|while|try|except|return|yield|with|as)/g, 
                '<span class="keyword" style="color: #569cd6; font-weight: bold;">$1</span>')
        .replace(/(True|False|None|self)/g, 
                '<span class="literal" style="color: #4fc1ff;">$1</span>')
        .replace(/(#.*)/g, 
                '<span class="comment" style="color: #6a9955; font-style: italic;">$1</span>')
        .replace(/('.*?'|".*?")/g, 
                '<span class="string" style="color: #ce9178;">$1</span>')
        .replace(/(\d+\.?\d*)/g, 
                '<span class="number" style="color: #b5cea8;">$1</span>');
    
    codeBlock.innerHTML = highlighted;
}

function selectText(element) {
    const range = document.createRange();
    range.selectNodeContents(element);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
}

function initializeHoverEffects(slideElement) {
    // Efectos hover para elementos interactivos
    const interactiveElements = slideElement.querySelectorAll('.stat-item, .dataset-card, .methodology-step');
    
    interactiveElements.forEach(element => {
        element.addEventListener('mouseenter', () => {
            element.style.transform = 'translateY(-3px)';
            element.style.boxShadow = '0 10px 25px rgba(3, 18, 64, 0.15)';
        });
        
        element.addEventListener('mouseleave', () => {
            element.style.transform = 'translateY(0)';
            element.style.boxShadow = '';
        });
    });
}

// ============================================================================
// DIAGRAMAS Y VISUALIZACIONES ESPECÍFICAS
// ============================================================================

function initializeCycleDiagram() {
    const cycleSteps = document.querySelectorAll('.cycle-step');
    
    cycleSteps.forEach((step, index) => {
        step.addEventListener('click', () => {
            activateCycleStep(step, cycleSteps, index);
        });
        
        step.addEventListener('mouseenter', () => {
            step.style.background = 'rgba(61, 157, 242, 0.1)';
        });
        
        step.addEventListener('mouseleave', () => {
            if (!step.classList.contains('active')) {
                step.style.background = '';
            }
        });
        
        setTimeout(() => {
            step.style.animation = 'cycle-step-appear 0.6s ease-out forwards';
        }, index * 200);
    });
    
    addCycleCSS();
}

function activateCycleStep(activeStep, allSteps, index) {
    allSteps.forEach(s => {
        s.classList.remove('active');
        s.style.background = '';
    });
    
    activeStep.classList.add('active');
    activeStep.style.background = 'linear-gradient(135deg, #0726D9, #3D9DF2)';
    activeStep.style.color = 'white';
    
    showCycleStepDetails(index + 1);
}

function showCycleStepDetails(stepNumber) {
    const stepNames = [
        'Adquisición de Datos',
        'Análisis y Preprocesamiento',
        'Extracción de Características',
        'Entrenamiento del Modelo',
        'Evaluación y Resultados'
    ];
    
    const details = [
        'Descarga y organización de datasets (RAVDESS, TESS, MESD). Total: 7,296 archivos de audio emocional.',
        'Limpieza, normalización de audio (22.05kHz) y preparación con Librosa para extracción de características.',
        'Cálculo de 180 características: MFCCs (40) + Chroma (12) + Mel-spectrograms (128) por cada archivo.',
        'CNN 1D con 47,175 parámetros, ADAM optimizer, 50 épocas de entrenamiento con validación.',
        'Validación: 89.73% precisión, métricas detalladas y análisis de matriz de confusión.'
    ];
    
    const message = `${stepNames[stepNumber - 1]}: ${details[stepNumber - 1]}`;
    showMessage(message, 6000);
    
    logInteraction('cycle_step_selected', currentSlide, { step: stepNumber });
}

function addCycleCSS() {
    if (document.querySelector('#cycle-animation-style')) return;
    
    const style = document.createElement('style');
    style.id = 'cycle-animation-style';
    style.textContent = `
        @keyframes cycle-step-appear {
            from {
                opacity: 0;
                transform: rotate(calc(72deg * var(--i))) translateY(-230px) rotate(calc(-72deg * var(--i))) scale(0.5);
            }
            to {
                opacity: 1;
                transform: rotate(calc(72deg * var(--i))) translateY(-230px) rotate(calc(-72deg * var(--i))) scale(1);
            }
        }
        
        @keyframes explanation-disappear {
            from { opacity: 1; transform: translateY(0); }
            to { opacity: 0; transform: translateY(-10px); }
        }
    `;
    document.head.appendChild(style);
}

function initializeInteractivePlots() {
    const iframes = document.querySelectorAll('.iframe-plot-container iframe');
    iframes.forEach((iframe, index) => {
        iframe.addEventListener('load', () => {
            console.log('Gráfico 3D cargado:', iframe.src);
            iframe.style.opacity = '0';
            iframe.style.transition = 'opacity 0.5s ease';
            setTimeout(() => iframe.style.opacity = '1', 100);
        });
        
        iframe.addEventListener('error', () => {
            console.warn('Error cargando gráfico:', iframe.src);
            const container = iframe.parentElement;
            if (container) {
                container.innerHTML = `
                    <div style="display: flex; align-items: center; justify-content: center; 
                                height: 400px; background: #f0f8ff; border-radius: 10px; 
                                border: 2px dashed #3D9DF2;">
                        <div style="text-align: center; color: #1760BF;">
                            <div style="font-size: 3rem; margin-bottom: 10px;">📊</div>
                            <div>Gráfico 3D no disponible</div>
                            <div style="font-size: 0.9rem; margin-top: 5px;">
                                Visualización interactiva de ${index === 0 ? 'PCA' : 'LDA'}
                            </div>
                        </div>
                    </div>
                `;
            }
        });
    });
}

function initializeArchitectureSummary() {
    const table = document.querySelector('.cnn-table');
    if (!table) return;
    
    const rows = table.querySelectorAll('tbody tr');
    rows.forEach((row, index) => {
        row.addEventListener('mouseenter', () => highlightArchitectureLayer(row, index));
        row.addEventListener('mouseleave', () => unhighlightArchitectureLayer(row, index));
        row.addEventListener('click', () => showArchitectureDetails(index));
    });
}

function highlightArchitectureLayer(row, layerIndex) {
    row.style.background = 'linear-gradient(90deg, rgba(61, 157, 242, 0.1), rgba(7, 38, 217, 0.05))';
    row.style.borderLeft = '4px solid #3D9DF2';
    row.style.transform = 'translateX(5px)';
    console.log(`Highlighting layer ${layerIndex}`);
}

function unhighlightArchitectureLayer(row, layerIndex) {
    row.style.background = '';
    row.style.borderLeft = '';
    row.style.transform = 'translateX(0)';
    console.log(`Unhighlighting layer ${layerIndex}`);
}

function showArchitectureDetails(layerIndex) {
    const architectureDetails = {
        0: 'Input Layer: Punto de entrada del modelo, recibe las 180 características extraídas del audio',
        1: 'Conv1D_1: Detecta patrones básicos con 128 filtros. Cada filtro aprende un patrón específico',
        2: 'BatchNorm_1: Estabiliza el entrenamiento normalizando las activaciones',
        3: 'MaxPool_1: Reduce la dimensionalidad espacial manteniendo características importantes',
        4: 'Dropout_1: Previene overfitting desactivando neuronas aleatoriamente',
        5: 'Conv1D_2: Detecta patrones más complejos combinando características de la capa anterior',
        6: 'BatchNorm_2: Segunda normalización para mantener estabilidad',
        7: 'MaxPool_2: Segunda reducción dimensional',
        8: 'Dropout_2: Regularización adicional',
        9: 'GlobalAvgPool: Condensa cada mapa de características en un solo valor',
        10: 'Dense_Inter: Procesamiento final de características antes de clasificación',
        11: 'Dropout_Final: Última regularización',
        12: 'Output: Clasificación final con Softmax para 7 emociones'
    };
    
    const detail = architectureDetails[layerIndex] || 'Capa de la arquitectura CNN 1D';
    showMessage(detail, 4000);
    logInteraction('architecture_layer_click', currentSlide, { layer: layerIndex });
}

function initializeConfusionMatrix() {
    const matrixImage = document.querySelector('img[alt*="Matriz de Confusión"], img[src*="confusion_matrix"]');
    if (matrixImage) {
        matrixImage.style.cursor = 'pointer';
        matrixImage.title = 'Click para vista ampliada';
        matrixImage.addEventListener('click', () => showImageModal(matrixImage));
        
        // Añadir efecto hover
        matrixImage.addEventListener('mouseenter', () => {
            matrixImage.style.transform = 'scale(1.02)';
            matrixImage.style.transition = 'transform 0.3s ease';
        });
        
        matrixImage.addEventListener('mouseleave', () => {
            matrixImage.style.transform = 'scale(1)';
        });
    }
}

function showImageModal(image) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.9); display: flex; justify-content: center;
        align-items: center; z-index: 10000; cursor: pointer;
        animation: modal-appear 0.3s ease-out;
    `;
    
    const modalImage = document.createElement('img');
    modalImage.src = image.src;
    modalImage.alt = image.alt;
    modalImage.style.cssText = `
        max-width: 90%; max-height: 90%; border-radius: 10px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        animation: image-zoom 0.3s ease-out;
    `;
    
    const closeButton = document.createElement('button');
    closeButton.innerHTML = '✕';
    closeButton.style.cssText = `
        position: absolute; top: 20px; right: 20px;
        background: white; border: none; border-radius: 50%;
        width: 40px; height: 40px; font-size: 20px; cursor: pointer;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    `;
    
    modal.appendChild(modalImage);
    modal.appendChild(closeButton);
    document.body.appendChild(modal);
    
    // Event listeners
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });
    
    closeButton.addEventListener('click', () => modal.remove());
    
    // Keyboard support
    const handleKeyDown = (e) => {
        if (e.key === 'Escape') {
            modal.remove();
            document.removeEventListener('keydown', handleKeyDown);
        }
    };
    document.addEventListener('keydown', handleKeyDown);
    
    // Añadir estilos de animación
    if (!document.querySelector('#modal-animations')) {
        const style = document.createElement('style');
        style.id = 'modal-animations';
        style.textContent = `
            @keyframes modal-appear {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            @keyframes image-zoom {
                from { transform: scale(0.8); opacity: 0; }
                to { transform: scale(1); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    logInteraction('image_modal_open', currentSlide, { image: image.alt });
}

// ============================================================================
// CONTROLES DE AUDIO
// ============================================================================

function updateAudioControls(slideNumber) {
    const currentSlide = document.querySelector(`[data-slide="${slideNumber}"]`);
    if (!currentSlide) return;
    
    // Pausar audios de otras diapositivas
    document.querySelectorAll('audio').forEach(audio => {
        if (!currentSlide.contains(audio)) {
            audio.pause();
            audio.currentTime = 0;
        }
    });
    
    // Configurar audios actuales
    const audioElements = currentSlide.querySelectorAll('audio');
    audioElements.forEach((audio, index) => {
        enhanceAudioControls(audio, index);
        addAudioEventListeners(audio, index, slideNumber);
    });
}

function enhanceAudioControls(audioElement, index) {
    audioElement.style.borderRadius = '25px';
    audioElement.style.boxShadow = '0 5px 15px rgba(61, 157, 242, 0.2)';
    audioElement.style.border = '2px solid #3D9DF2';
    audioElement.title = `Audio de muestra ${index + 1} - Click para reproducir`;
    
    // Añadir controles personalizados si es necesario
    audioElement.controls = true;
    audioElement.preload = 'metadata';
}

function addAudioEventListeners(audio, index, slideNumber) {
    audio.addEventListener('play', () => {
        logInteraction('audio_play', slideNumber, { audio: index });
        console.log(`Audio ${index + 1} iniciado`);
    });
    
    audio.addEventListener('pause', () => {
        logInteraction('audio_pause', slideNumber, { audio: index });
        console.log(`Audio ${index + 1} pausado`);
    });
    
    audio.addEventListener('ended', () => {
        logInteraction('audio_ended', slideNumber, { audio: index });
        console.log(`Audio ${index + 1} terminado`);
    });
    
    audio.addEventListener('error', () => {
        console.warn(`Error cargando audio ${index + 1}`);
        // Mostrar mensaje de error amigable
        const errorMessage = document.createElement('div');
        errorMessage.style.cssText = `
            background: #f8d7da; color: #721c24; padding: 10px;
            border-radius: 5px; margin: 10px 0; text-align: center;
        `;
        errorMessage.textContent = `Audio ${index + 1} no disponible`;
        audio.parentNode.insertBefore(errorMessage, audio.nextSibling);
    });
    
    audio.addEventListener('loadstart', () => {
        console.log(`Cargando audio ${index + 1}...`);
    });
    
    audio.addEventListener('canplay', () => {
        console.log(`Audio ${index + 1} listo para reproducir`);
    });
}

// ============================================================================
// NAVEGACIÓN POR TECLADO
// ============================================================================

function handleKeyboardNavigation(e) {
    if (!config.keyboardShortcuts) return;
    
    // Ignorar si hay elementos focusados
    if (isInputFocused()) return;
    
    const keyActions = {
        'ArrowRight': () => nextSlide(),
        'ArrowDown': () => nextSlide(),
        ' ': () => nextSlide(),
        'PageDown': () => nextSlide(),
        'ArrowLeft': () => previousSlide(),
        'ArrowUp': () => previousSlide(),
        'PageUp': () => previousSlide(),
        'Home': () => jumpToSlide(1, 'keyboard_home'),
        'End': () => jumpToSlide(totalSlides, 'keyboard_end'),
        'Escape': () => { exitFullscreen(); exitPresentationMode(); },
        'b': () => toggleBookmark(currentSlide),
        'B': () => toggleBookmark(currentSlide),
        'h': () => showHelpModal(),
        'H': () => showHelpModal(),
        '?': () => showHelpModal()
    };
    
    // Atajos con Ctrl/Cmd
    if (e.ctrlKey || e.metaKey) {
        const ctrlActions = {
            'f': () => toggleFullscreen(),
            'F': () => toggleFullscreen(),
            'p': () => togglePresentationMode(),
            'P': () => togglePresentationMode(),
            'a': () => toggleAutoAdvance(),
            'A': () => toggleAutoAdvance(),
            's': () => exportAnalytics(),
            'S': () => exportAnalytics()
        };
        
        if (ctrlActions[e.key]) {
            e.preventDefault();
            ctrlActions[e.key]();
            return;
        }
        
        // Navegación numérica
        const num = parseInt(e.key);
        if (num >= 1 && num <= 9) {
            e.preventDefault();
            jumpToSlide(num, 'keyboard_number');
            return;
        }
        if (e.key === '0') {
            e.preventDefault();
            jumpToSlide(10, 'keyboard_number');
            return;
        }
    }
    
    // Navegación por números sin Ctrl (para diapositivas 1-9)
    if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        const num = parseInt(e.key);
        if (num >= 1 && num <= 9) {
            e.preventDefault();
            jumpToSlide(num, 'keyboard_direct');
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
// ANALYTICS Y TRACKING - CORREGIDO
// ============================================================================

function trackSlideView(slideNumber, source) {
    if (!config.analytics) return;
    
    const now = Date.now();
    const views = presentationState.analytics.slideViews;
    
    if (!views.has(slideNumber)) {
        views.set(slideNumber, {
            firstView: now,
            totalTime: 0,
            viewCount: 0,
            sources: new Set() // CORREGIDO: Inicializar como Set
        });
    }
    
    const slideData = views.get(slideNumber);
    slideData.viewCount++;
    slideData.sources.add(source); // Ahora funciona correctamente
    slideData.lastView = now;
    
    // Calcular tiempo en diapositiva anterior
    updatePreviousSlideTime(now);
    
    presentationState.analytics.lastSlide = slideNumber;
    presentationState.analytics.lastSlideTime = now;
}

function updatePreviousSlideTime(now) {
    if (presentationState.analytics.lastSlideTime && presentationState.analytics.lastSlide) {
        const previousSlide = presentationState.analytics.lastSlide;
        const timeSpent = now - presentationState.analytics.lastSlideTime;
        
        const views = presentationState.analytics.slideViews;
        if (views.has(previousSlide)) {
            views.get(previousSlide).totalTime += timeSpent;
        }
    }
}

function logInteraction(type, slideNumber, data = {}) {
    if (!config.analytics) return;
    
    const interaction = {
        type,
        slideNumber,
        timestamp: Date.now(),
        data
    };
    
    presentationState.analytics.interactions.push(interaction);
    
    // Limitar el número de interacciones almacenadas
    if (presentationState.analytics.interactions.length > 1000) {
        presentationState.analytics.interactions = presentationState.analytics.interactions.slice(-500);
    }
    
    // Analytics externos (Google Analytics si está disponible)
    if (typeof gtag !== 'undefined') {
        gtag('event', type, {
            slide_number: slideNumber,
            custom_parameter_1: JSON.stringify(data)
        });
    }
    
    console.log('📊 Interaction:', interaction);
}

function generateAnalyticsReport() {
    const analytics = presentationState.analytics;
    const totalTime = Date.now() - analytics.startTime;
    const slideViews = Array.from(analytics.slideViews.entries());
    
    return {
        sessionInfo: {
            startTime: new Date(analytics.startTime).toISOString(),
            endTime: new Date().toISOString(),
            totalDuration: totalTime,
            totalDurationFormatted: formatDuration(totalTime)
        },
        slideStats: {
            totalSlides: totalSlides,
            slidesViewed: slideViews.length,
            currentSlide: currentSlide,
            completionRate: Math.round((slideViews.length / totalSlides) * 100),
            averageTimePerSlide: slideViews.length > 0 ? 
                Math.round(slideViews.reduce((sum, [_, data]) => sum + data.totalTime, 0) / slideViews.length) : 0
        },
        topSlides: {
            mostViewed: slideViews
                .sort(([,a], [,b]) => b.viewCount - a.viewCount)
                .slice(0, 5)
                .map(([slide, data]) => ({ 
                    slide, 
                    views: data.viewCount,
                    totalTime: formatDuration(data.totalTime)
                })),
            longestViewed: slideViews
                .sort(([,a], [,b]) => b.totalTime - a.totalTime)
                .slice(0, 5)
                .map(([slide, data]) => ({ 
                    slide, 
                    time: formatDuration(data.totalTime),
                    views: data.viewCount
                }))
        },
        interactions: {
            totalCount: analytics.interactions.length,
            byType: analytics.interactions.reduce((acc, int) => {
                acc[int.type] = (acc[int.type] || 0) + 1;
                return acc;
            }, {}),
            recentInteractions: analytics.interactions.slice(-10)
        },
        navigation: {
            bookmarkedSlides: Array.from(presentationState.bookmarks),
            navigationHistory: presentationState.slideHistory.slice(-20),
            backtrackingRate: calculateBacktrackingRate()
        }
    };
}

function calculateBacktrackingRate() {
    const history = presentationState.slideHistory;
    if (history.length < 2) return 0;
    
    let backtrackCount = 0;
    for (let i = 1; i < history.length; i++) {
        if (history[i] < history[i-1]) {
            backtrackCount++;
        }
    }
    
    return Math.round((backtrackCount / (history.length - 1)) * 100);
}

function formatDuration(milliseconds) {
    const seconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

function exportAnalytics() {
    const report = generateAnalyticsReport();
    const dataStr = JSON.stringify(report, null, 2);
    const dataBlob = new Blob([dataStr], {type: 'application/json'});
    
    const link = document.createElement('a');
    link.href = URL.createObjectURL(dataBlob);
    link.download = `cnn-presentation-analytics-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    
    showMessage('Analytics exportados exitosamente', 2000);
    logInteraction('analytics_export', currentSlide);
    
    // También mostrar resumen en consola
    console.log('📊 Analytics Summary:', report);
}

function showAnalyticsSummary() {
    const report = generateAnalyticsReport();
    const summaryModal = document.createElement('div');
    summaryModal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8); display: flex; justify-content: center;
        align-items: center; z-index: 10000; backdrop-filter: blur(5px);
    `;
    
    const content = document.createElement('div');
    content.style.cssText = `
        background: white; border-radius: 15px; padding: 30px;
        max-width: 600px; max-height: 80vh; overflow-y: auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    `;
    
    content.innerHTML = `
        <h2 style="color: #0726D9; margin-bottom: 20px;">📊 Resumen de Analytics</h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div>
                <h3>⏱️ Tiempo de Sesión</h3>
                <p><strong>${report.sessionInfo.totalDurationFormatted}</strong></p>
            </div>
            <div>
                <h3>📈 Progreso</h3>
                <p><strong>${report.slideStats.completionRate}%</strong> completado</p>
            </div>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h3>🏆 Diapositivas Más Vistas</h3>
            <ol>
                ${report.topSlides.mostViewed.map(slide => 
                    `<li>Slide ${slide.slide}: ${slide.views} vistas</li>`
                ).join('')}
            </ol>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h3>🔍 Interacciones</h3>
            <p>Total: <strong>${report.interactions.totalCount}</strong></p>
            <p>Marcadores: <strong>${report.navigation.bookmarkedSlides.length}</strong></p>
        </div>
        
        <div style="text-align: center; margin-top: 25px;">
            <button onclick="this.closest('div').parentElement.remove()" 
                    style="background: #0726D9; color: white; border: none; padding: 10px 20px; 
                           border-radius: 8px; cursor: pointer; margin-right: 10px;">
                Cerrar
            </button>
            <button onclick="exportAnalytics(); this.closest('div').parentElement.remove();" 
                    style="background: #28a745; color: white; border: none; padding: 10px 20px; 
                           border-radius: 8px; cursor: pointer;">
                Exportar Completo
            </button>
        </div>
    `;
    
    summaryModal.appendChild(content);
    document.body.appendChild(summaryModal);
    
    logInteraction('analytics_summary_view', currentSlide);
}

// ============================================================================
// MODOS DE PRESENTACIÓN
// ============================================================================

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().then(() => {
            presentationState.mode = 'fullscreen';
            showMessage('Modo Pantalla Completa Activado', 2000);
            logInteraction('fullscreen_enter', currentSlide);
        }).catch(err => {
            console.log('Error enabling fullscreen:', err);
            showMessage('No se pudo activar pantalla completa', 2000);
        });
    } else {
        document.exitFullscreen().then(() => {
            presentationState.mode = 'normal';
            showMessage('Pantalla Completa Desactivada', 2000);
            logInteraction('fullscreen_exit', currentSlide);
        });
    }
}

function exitFullscreen() {
    if (document.fullscreenElement) {
        document.exitFullscreen();
    }
}

function togglePresentationMode() {
    if (presentationState.mode === 'presentation') {
        exitPresentationMode();
    } else {
        enterPresentationMode();
    }
}

function enterPresentationMode() {
    presentationState.mode = 'presentation';
    document.body.classList.add('presentation-mode');
    
    hideUIElements();
    showMessage('Modo Presentación Activado\nUsa Esc para salir', 3000);
    logInteraction('presentation_mode_enter', currentSlide);
}

function exitPresentationMode() {
    presentationState.mode = 'normal';
    document.body.classList.remove('presentation-mode');
    
    showUIElements();
    showMessage('Modo Presentación Desactivado', 2000);
    logInteraction('presentation_mode_exit', currentSlide);
}

function hideUIElements() {
    const navigation = document.querySelector('.navigation');
    const counter = document.querySelector('.slide-counter');
    
    if (navigation) navigation.style.display = 'none';
    if (counter) counter.style.display = 'none';
}

function showUIElements() {
    const navigation = document.querySelector('.navigation');
    const counter = document.querySelector('.slide-counter');
    
    if (navigation) navigation.style.display = 'flex';
    if (counter) counter.style.display = 'block';
}

function toggleAutoAdvance() {
    if (presentationState.autoAdvance) {
        stopAutoAdvance();
    } else {
        startAutoAdvance();
    }
}

function startAutoAdvance() {
    presentationState.autoAdvance = true;
    
    const advance = () => {
        if (currentSlide < totalSlides) {
            nextSlide();
            presentationState.autoAdvanceInterval = setTimeout(advance, config.autoAdvanceDelay);
        } else {
            stopAutoAdvance();
            showMessage('Auto-avance completado', 2000);
        }
    };
    
    presentationState.autoAdvanceInterval = setTimeout(advance, config.autoAdvanceDelay);
    showMessage(`Auto-avance activado (${config.autoAdvanceDelay/1000}s por diapositiva)`, 3000);
    logInteraction('auto_advance_start', currentSlide);
}

function stopAutoAdvance() {
    presentationState.autoAdvance = false;
    if (presentationState.autoAdvanceInterval) {
        clearTimeout(presentationState.autoAdvanceInterval);
        presentationState.autoAdvanceInterval = null;
    }
    showMessage('Auto-avance desactivado', 2000);
    logInteraction('auto_advance_stop', currentSlide);
}

// ============================================================================
// SISTEMA DE MARCADORES
// ============================================================================

function toggleBookmark(slideNumber = currentSlide) {
    if (presentationState.bookmarks.has(slideNumber)) {
        presentationState.bookmarks.delete(slideNumber);
        showMessage(`Marcador eliminado de diapositiva ${slideNumber}`, 2000);
    } else {
        presentationState.bookmarks.add(slideNumber);
        showMessage(`Marcador añadido a diapositiva ${slideNumber}`, 2000);
    }
    
    updateBookmarkIndicators();
    logInteraction('bookmark_toggle', slideNumber, { 
        bookmarked: presentationState.bookmarks.has(slideNumber) 
    });
}

function updateBookmarkIndicators() {
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => {
        const slideNum = parseInt(slide.dataset.slide);
        let indicator = slide.querySelector('.bookmark-indicator');
        
        if (presentationState.bookmarks.has(slideNum)) {
            if (!indicator) {
                indicator = createBookmarkIndicator();
                slide.appendChild(indicator);
            }
            indicator.style.display = 'block';
        } else if (indicator) {
            indicator.style.display = 'none';
        }
    });
}

function createBookmarkIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'bookmark-indicator';
    indicator.style.cssText = `
        position: absolute; top: 10px; left: 10px;
        background: #FFD700; color: #333; padding: 5px 10px;
        border-radius: 15px; font-size: 0.8rem; font-weight: bold;
        z-index: 100; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        animation: bookmark-appear 0.3s ease-out;
    `;
    indicator.innerHTML = '📌 Marcado';
    
    // Añadir animación de marcador
    if (!document.querySelector('#bookmark-style')) {
        const style = document.createElement('style');
        style.id = 'bookmark-style';
        style.textContent = `
            @keyframes bookmark-appear {
                from { opacity: 0; transform: scale(0.5) rotate(-10deg); }
                to { opacity: 1; transform: scale(1) rotate(0deg); }
            }
        `;
        document.head.appendChild(style);
    }
    
    return indicator;
}

// ============================================================================
// SISTEMA DE AYUDA
// ============================================================================

function showHelpModal() {
    const modal = createHelpModal();
    document.body.appendChild(modal);
    
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });
    
    // Soporte de teclado
    const handleKeyDown = (e) => {
        if (e.key === 'Escape') {
            modal.remove();
            document.removeEventListener('keydown', handleKeyDown);
        }
    };
    document.addEventListener('keydown', handleKeyDown);
    
    logInteraction('help_modal_open', currentSlide);
}

function createHelpModal() {
    const modal = document.createElement('div');
    modal.id = 'help-modal';
    modal.style.cssText = `
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8); display: flex; justify-content: center;
        align-items: center; z-index: 10000; backdrop-filter: blur(5px);
        animation: modal-appear 0.3s ease-out;
    `;
    
    const helpContent = document.createElement('div');
    helpContent.style.cssText = `
        background: white; border-radius: 15px; padding: 30px;
        max-width: 700px; max-height: 80vh; overflow-y: auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        animation: help-content-appear 0.4s ease-out;
    `;
    
    helpContent.innerHTML = getHelpContent();
    modal.appendChild(helpContent);
    
    return modal;
}

function getHelpContent() {
    return `
        <h2 style="color: #0726D9; margin-bottom: 20px; text-align: center;">
            🎯 Guía de Navegación - CNN 1D Presentation
        </h2>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
            <div>
                <h3 style="color: #1760BF; margin-bottom: 15px;">⌨️ Atajos de Teclado</h3>
                <div style="font-family: monospace; background: #f5f5f5; padding: 15px; border-radius: 8px;">
                    <strong>Navegación Básica:</strong><br>
                    • → / ↓ / Espacio: Siguiente<br>
                    • ← / ↑: Anterior<br>
                    • Home: Primera diapositiva<br>
                    • End: Última diapositiva<br>
                    • 1-9: Ir a diapositiva específica<br><br>
                    
                    <strong>Modos Especiales:</strong><br>
                    • Ctrl+F: Pantalla completa<br>
                    • Ctrl+P: Modo presentación<br>
                    • Esc: Salir de modos<br>
                    • Ctrl+A: Auto-avance<br><br>
                    
                    <strong>Utilidades:</strong><br>
                    • B: Marcar diapositiva<br>
                    • H / ?: Mostrar ayuda<br>
                    • Ctrl+S: Exportar analytics
                </div>
            </div>
            
            <div>
                <h3 style="color: #1760BF; margin-bottom: 15px;">📱 Controles Táctiles</h3>
                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px;">
                    • Deslizar ←: Siguiente<br>
                    • Deslizar →: Anterior<br>
                    • Tap: Interacciones específicas<br>
                    • Pellizcar: Zoom en imágenes<br><br>
                    
                    <strong>Elementos Interactivos:</strong><br>
                    • Tablas: Click en filas<br>
                    • Fórmulas: Click para explicar<br>
                    • Código: Click para copiar<br>
                    • Imágenes: Click para ampliar
                </div>
            </div>
        </div>
        
        <h3 style="color: #1760BF; margin-top: 20px;">🎛️ Características Técnicas</h3>
        <div style="background: #fff8f0; padding: 15px; border-radius: 8px; margin: 10px 0;">
            • <strong>Capas CNN:</strong> Click para ver detalles técnicos<br>
            • <strong>Diagramas:</strong> Elementos interactivos con animaciones<br>
            • <strong>Gráficos 3D:</strong> PCA y LDA interactivos (si disponibles)<br>
            • <strong>Audio:</strong> Controles mejorados con tracking<br>
            • <strong>Marcadores:</strong> Sistema de bookmarks persistente<br>
            • <strong>Analytics:</strong> Seguimiento completo de interacciones
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 25px;">
            <div style="font-size: 0.9rem; color: #666;">
                Diapositiva actual: ${currentSlide} / ${totalSlides}
            </div>
            <button onclick="document.getElementById('help-modal').remove()" 
                    style="background: #0726D9; color: white; border: none; padding: 10px 20px; 
                           border-radius: 8px; cursor: pointer; font-weight: bold;">
                Cerrar Ayuda
            </button>
        </div>
    `;
}

// ============================================================================
// GESTIÓN TÁCTIL
// ============================================================================

function initializeTouchHandlers() {
    if (!config.touchGestures) return;
    
    document.addEventListener('touchstart', handleTouchStart, { passive: false });
    document.addEventListener('touchmove', handleTouchMove, { passive: false });
    document.addEventListener('touchend', handleTouchEnd, { passive: false });
}

function handleTouchStart(e) {
    isTouching = true;
    const touch = e.touches[0];
    touchStartX = touch.clientX;
    touchStartY = touch.clientY;
    
    logInteraction('touch_start', currentSlide, {
        x: touchStartX,
        y: touchStartY
    });
}

function handleTouchMove(e) {
    if (!isTouching) return;
    
    const touch = e.touches[0];
    const deltaX = touch.clientX - touchStartX;
    const deltaY = touch.clientY - touchStartY;
    
    // Prevenir scroll durante gesture horizontal
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 20) {
        e.preventDefault();
    }
}

function handleTouchEnd(e) {
    if (!isTouching) return;
    
    isTouching = false;
    const touch = e.changedTouches[0];
    touchEndX = touch.clientX;
    touchEndY = touch.clientY;
    
    handleSwipeGesture();
}

function handleSwipeGesture() {
    const deltaX = touchEndX - touchStartX;
    const deltaY = touchEndY - touchStartY;
    const minSwipeDistance = 50;
    const maxVerticalDeviation = 100;
    
    if (Math.abs(deltaX) > minSwipeDistance && Math.abs(deltaY) < maxVerticalDeviation) {
        if (deltaX > 0) {
            previousSlide();
            logInteraction('touch_swipe_previous', currentSlide);
        } else {
            nextSlide();
            logInteraction('touch_swipe_next', currentSlide);
        }
    }
}

// ============================================================================
// UTILIDADES Y GESTIÓN DE SESIÓN - CORREGIDO
// ============================================================================

function preloadNearbySlides(currentSlideNum) {
    const preloadSlides = [currentSlideNum - 1, currentSlideNum + 1];
    
    preloadSlides.forEach(slideNum => {
        if (slideNum >= 1 && slideNum <= totalSlides) {
            preloadSlideImages(slideNum);
        }
    });
}

function preloadSlideImages(slideNum) {
    const slide = document.querySelector(`[data-slide="${slideNum}"]`);
    if (!slide) return;
    
    const images = slide.querySelectorAll('img');
    images.forEach(img => {
        if (img.src && !img.complete) {
            const preloadImg = new Image();
            preloadImg.src = img.src;
            preloadImg.onerror = () => {
                console.warn(`Failed to preload image: ${img.src}`);
            };
        }
    });
}

function handleHashChange() {
    const hash = window.location.hash;
    if (hash.startsWith('#slide-')) {
        const slideNumber = parseInt(hash.substring(7));
        if (slideNumber >= 1 && slideNumber <= totalSlides && slideNumber !== currentSlide) {
            currentSlide = slideNumber;
            showSlide(currentSlide, 'url_hash');
        }
    }
}

function saveSession() {
    if (!localStorage) return;
    
    try {
        const sessionData = {
            currentSlide,
            bookmarks: Array.from(presentationState.bookmarks),
            analytics: {
                slideViews: Array.from(presentationState.analytics.slideViews.entries()).map(([slide, data]) => [
                    slide, 
                    {
                        ...data,
                        sources: Array.from(data.sources) // Convertir Set a Array para serialización
                    }
                ]),
                interactions: presentationState.analytics.interactions.slice(-100), // Solo últimas 100
                startTime: presentationState.analytics.startTime
            },
            timestamp: Date.now(),
            version: '1.0'
        };
        
        localStorage.setItem('cnn-presentation-session', JSON.stringify(sessionData));
        console.log('💾 Sesión guardada');
    } catch (error) {
        console.warn('No se pudo guardar la sesión:', error);
    }
}

function loadSession() {
    if (!localStorage) return false;
    
    try {
        const sessionData = JSON.parse(localStorage.getItem('cnn-presentation-session'));
        if (!sessionData) return false;
        
        // Solo restaurar sesiones recientes (< 24h)
        const hoursSinceSession = (Date.now() - sessionData.timestamp) / (1000 * 60 * 60);
        if (hoursSinceSession > 24) {
            localStorage.removeItem('cnn-presentation-session');
            return false;
        }
        
        // Restaurar estado
        if (sessionData.currentSlide >= 1 && sessionData.currentSlide <= totalSlides) {
            currentSlide = sessionData.currentSlide;
        }
        
        if (sessionData.bookmarks && Array.isArray(sessionData.bookmarks)) {
            presentationState.bookmarks = new Set(sessionData.bookmarks);
        }
        
        if (sessionData.analytics?.slideViews && Array.isArray(sessionData.analytics.slideViews)) {
            // Restaurar slideViews y convertir sources de Array a Set
            const restoredViews = new Map();
            sessionData.analytics.slideViews.forEach(([slide, data]) => {
                if (data && typeof data === 'object') {
                    restoredViews.set(slide, {
                        firstView: data.firstView || Date.now(),
                        totalTime: data.totalTime || 0,
                        viewCount: data.viewCount || 0,
                        lastView: data.lastView || Date.now(),
                        sources: new Set(Array.isArray(data.sources) ? data.sources : []) // Convertir Array de vuelta a Set
                    });
                }
            });
            presentationState.analytics.slideViews = restoredViews;
        }
        
        if (sessionData.analytics?.interactions && Array.isArray(sessionData.analytics.interactions)) {
            presentationState.analytics.interactions = sessionData.analytics.interactions;
        }
        
        if (sessionData.analytics?.startTime) {
            presentationState.analytics.startTime = sessionData.analytics.startTime;
        }
        
        console.log(`📁 Sesión restaurada: diapositiva ${currentSlide}`);
        return true;
    } catch (error) {
        console.warn('Error cargando sesión:', error);
        localStorage.removeItem('cnn-presentation-session');
        return false;
    }
}

function validateSlideContent() {
    const slides = document.querySelectorAll('.slide');
    const issues = [];
    
    slides.forEach((slide, index) => {
        const slideNumber = index + 1;
        
        // Verificar título
        if (!slide.querySelector('h1, h2')) {
            issues.push(`Slide ${slideNumber}: Sin título`);
        }
        
        // Verificar imágenes
        const images = slide.querySelectorAll('img');
        images.forEach((img, imgIndex) => {
            img.onerror = () => {
                issues.push(`Slide ${slideNumber}: Imagen ${imgIndex + 1} rota - ${img.src}`);
            };
        });
        
        // Verificar audio
        const audios = slide.querySelectorAll('audio');
        audios.forEach((audio, audioIndex) => {
            audio.onerror = () => {
                issues.push(`Slide ${slideNumber}: Audio ${audioIndex + 1} no disponible`);
            };
        });
    });
    
    if (issues.length > 0) {
        console.warn('⚠️ Problemas encontrados:', issues);
    } else {
        console.log('✅ Validación exitosa - Todas las diapositivas están correctas');
    }
    
    return issues;
}

// ============================================================================
// OPTIMIZACIÓN DE RENDIMIENTO
// ============================================================================

function optimizePerformance() {
    // Lazy loading para imágenes
    if ('IntersectionObserver' in window) {
        setupLazyLoading();
    }
    
    // Debounce para resize
    setupResizeHandler();
    
    // Optimizar MathJax si está disponible
    if (typeof MathJax !== 'undefined') {
        optimizeMathJax();
    }
}

function setupLazyLoading() {
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    imageObserver.unobserve(img);
                }
            }
        });
    }, {
        rootMargin: '100px' // Cargar imágenes 100px antes de que sean visibles
    });
    
    const images = document.querySelectorAll('img');
    images.forEach(img => {
        // Solo aplicar lazy loading a imágenes que no están en la diapositiva actual
        if (img.src && !img.closest(`[data-slide="${currentSlide}"]`)) {
            img.dataset.src = img.src;
            img.src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100%" height="100%" fill="#f0f0f0"/></svg>';
            imageObserver.observe(img);
        }
    });
}

function setupResizeHandler() {
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(handleResize, 250);
    });
}

function handleResize() {
    // Ajustar iframes
    const iframes = document.querySelectorAll('.iframe-plot-container iframe');
    iframes.forEach(iframe => {
        const container = iframe.parentElement;
        if (container) {
            const containerWidth = container.offsetWidth;
            iframe.style.height = `${Math.max(containerWidth * 0.5625, 300)}px`;
        }
    });
    
    // Reajustar elementos de ciclo si existen
    const cycleDiagram = document.querySelector('.iterative-cycle-diagram');
    if (cycleDiagram) {
        const viewportWidth = window.innerWidth;
        if (viewportWidth < 768) {
            cycleDiagram.style.transform = 'scale(0.8)';
        } else {
            cycleDiagram.style.transform = 'scale(1)';
        }
    }
    
    logInteraction('window_resize', currentSlide, {
        width: window.innerWidth,
        height: window.innerHeight
    });
}

function optimizeMathJax() {
    if (typeof MathJax !== 'undefined') {
        // Configurar MathJax para mejor rendimiento
    MathJax.config.tex.inlineMath = [['$', '$'], ['\\(', '\\)']];
    MathJax.config.tex.displayMath = [['$$', '$$'], ['\\[', '\\]']];
    MathJax.config.startup.ready = () => {
        console.log('🔢 MathJax optimizado y listo');
    };
    }
}

// ============================================================================
// EVENTOS DEL SISTEMA
// ============================================================================

function setupEventListeners() {
    // Navegación por teclado
    document.addEventListener('keydown', handleKeyboardNavigation);
    
    // Cambios de URL
    window.addEventListener('hashchange', handleHashChange);
    
    // Guardar sesión antes de cerrar
    window.addEventListener('beforeunload', saveSession);
    window.addEventListener('pagehide', saveSession);
    
    // Guardar periódicamente
    setInterval(saveSession, 30000); // Cada 30 segundos
    
    // Detectar cambios de visibilidad
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Detectar cambios de fullscreen
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    
    // Detectar errores globales
    window.addEventListener('error', handleGlobalError);
    window.addEventListener('unhandledrejection', handleUnhandledRejection);
    
    console.log('🎛️ Event listeners configurados');
}

function handleVisibilityChange() {
    if (document.hidden) {
        // Pausar auto-avance si está activo
        if (presentationState.autoAdvance) {
            stopAutoAdvance();
        }
        
        // Guardar sesión
        saveSession();
        
        // Pausar todos los audios
        document.querySelectorAll('audio').forEach(audio => audio.pause());
        
        console.log('👁️ Presentación oculta - estado guardado');
    } else {
        console.log('👁️ Presentación visible');
    }
    
    logInteraction('visibility_change', currentSlide, { 
        hidden: document.hidden 
    });
}

function handleFullscreenChange() {
    const isFullscreen = !!document.fullscreenElement;
    
    if (isFullscreen) {
        console.log('🖥️ Modo pantalla completa activado');
    } else {
        console.log('🖥️ Modo pantalla completa desactivado');
    }
    
    logInteraction('fullscreen_change', currentSlide, { 
        fullscreen: isFullscreen 
    });
}

function handleGlobalError(event) {
    console.error('❌ Error global:', event.error);
    logInteraction('global_error', currentSlide, {
        message: event.error?.message,
        filename: event.filename,
        lineno: event.lineno
    });
}

function handleUnhandledRejection(event) {
    console.error('❌ Promise rechazada:', event.reason);
    logInteraction('unhandled_rejection', currentSlide, {
        reason: event.reason?.toString()
    });
}

// ============================================================================
// API PÚBLICA
// ============================================================================

function createPublicAPI() {
    return {
        // Navegación
        jumpToSlide,
        nextSlide,
        previousSlide,
        getCurrentSlide: () => currentSlide,
        getTotalSlides: () => totalSlides,
        
        // Modos
        toggleFullscreen,
        togglePresentationMode,
        toggleAutoAdvance,
        
        // Marcadores
        toggleBookmark,
        getBookmarks: () => Array.from(presentationState.bookmarks),
        clearAllBookmarks: () => {
            presentationState.bookmarks.clear();
            updateBookmarkIndicators();
            showMessage('Todos los marcadores eliminados', 2000);
        },
        
        // Analytics
        generateAnalyticsReport,
        exportAnalytics,
        showAnalyticsSummary,
        getAnalytics: () => ({ ...presentationState.analytics }),
        clearAnalytics: () => {
            presentationState.analytics.interactions = [];
            presentationState.analytics.slideViews.clear();
            showMessage('Analytics limpiados', 2000);
        },
        
        // Utilidades
        showHelp: showHelpModal,
        validateContent: validateSlideContent,
        saveSession,
        loadSession,
        
        // Estado
        getState: () => ({ 
            ...presentationState, 
            currentSlide,
            totalSlides 
        }),
        getConfig: () => ({ ...config }),
        
        // Configuración
        setConfig: (newConfig) => {
            Object.assign(config, newConfig);
            console.log('⚙️ Configuración actualizada:', config);
            logInteraction('config_updated', currentSlide, newConfig);
        }
    };
}

// ============================================================================
// INICIALIZACIÓN PRINCIPAL
// ============================================================================

function initializePresentation() {
    console.log('🚀 Inicializando presentación CNN 1D...');
    
    try {
        // Cargar sesión previa si existe
        const sessionRestored = loadSession();
        
        // Manejar URL hash
        handleHashChange();
        
        // Mostrar diapositiva inicial
        showSlide(currentSlide, 'initialization');
        
        // Configurar event listeners
        setupEventListeners();
        
        // Inicializar controles táctiles
        initializeTouchHandlers();
        
        // Actualizar marcadores
        updateBookmarkIndicators();
        
        // Optimizar rendimiento
        optimizePerformance();
        
        // Crear API pública
        window.PresentationAPI = createPublicAPI();
        
        // Validar contenido después de un delay
        setTimeout(validateSlideContent, 1000);
        
        console.log('✅ Presentación inicializada correctamente');
        console.log(`📊 Diapositiva: ${currentSlide}/${totalSlides}`);
        console.log(`🔖 Marcadores: ${presentationState.bookmarks.size}`);
        console.log(`🎛️ API pública disponible como window.PresentationAPI`);
        
        logInteraction('presentation_initialized', currentSlide, {
            sessionRestored,
            totalSlides,
            bookmarks: presentationState.bookmarks.size,
            userAgent: navigator.userAgent,
            screenResolution: `${screen.width}x${screen.height}`,
            viewportSize: `${window.innerWidth}x${window.innerHeight}`
        });
        
    } catch (error) {
        console.error('❌ Error inicializando presentación:', error);
        showMessage('Error al inicializar la presentación. Algunas funciones pueden no estar disponibles.', 4000);
        
        // Aún así, intentar crear una API básica
        window.PresentationAPI = {
            jumpToSlide: (n) => jumpToSlide(n, 'api_fallback'),
            nextSlide,
            previousSlide,
            getCurrentSlide: () => currentSlide,
            showHelp: showHelpModal
        };
    }
}

// ============================================================================
// AUTO-INICIALIZACIÓN Y EXPOSICIÓN GLOBAL
// ============================================================================

// Exponer funciones principales al scope global para compatibilidad con HTML
window.nextSlide = nextSlide;
window.previousSlide = previousSlide;
window.jumpToSlide = jumpToSlide;
window.showSlide = showSlide;

// Inicializar cuando el DOM esté listo
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializePresentation);
} else {
    initializePresentation();
}

// Exportar para entornos que lo soporten
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        initializePresentation,
        showSlide,
        nextSlide,
        previousSlide,
        jumpToSlide
    };
}

// Mensaje de consola final
console.log(`
🎯 CNN 1D Presentation System - LISTO
=====================================
📊 Total Slides: ${totalSlides}
⌨️  Keyboard shortcuts: HABILITADO
📱 Touch gestures: HABILITADO  
🔍 Interactive elements: LISTO
📈 Analytics: ACTIVO
🎛️ Public API: window.PresentationAPI

🚀 CONTROLES PRINCIPALES:
• ← → : Navegación básica
• H o ? : Ayuda completa
• B : Marcar diapositiva
• Ctrl+F : Pantalla completa

🔧 Para desarrolladores:
• Ctrl+S : Exportar analytics
`);