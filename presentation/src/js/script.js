// This file contains the JavaScript functionality for the presentation.
// It handles slide navigation, progress bar updates, and keyboard controls.

let currentSlide = 1;
let totalSlides = 17; 

function showSlide(n) {
    // Hide all slides
    const slides = document.querySelectorAll('.slide');
    slides.forEach(slide => slide.classList.add('hidden'));
    
    // Show current slide
    const current = document.querySelector(`[data-slide="${n}"]`);
    if (current) {
        current.classList.remove('hidden');
    }
    
    // Update counter
    document.getElementById('slideCounter').textContent = `${n} / ${totalSlides}`;
    
    // Update progress bar
    const progress = (n / totalSlides) * 100;
    document.getElementById('progressFill').style.width = `${progress}%`;
}

function nextSlide() {
    if (currentSlide < totalSlides) {
        currentSlide++;
        showSlide(currentSlide);
    }
}

function previousSlide() {
    if (currentSlide > 1) {
        currentSlide--;
        showSlide(currentSlide);
    }
}

// Keyboard navigation
document.addEventListener('keydown', function(e) {
    if (e.key === 'ArrowRight' || e.key === ' ') {
        nextSlide();
        e.preventDefault();
    } else if (e.key === 'ArrowLeft') {
        previousSlide();
        e.preventDefault();
    }
});

// Initialize
showSlide(1);

// Auto-advance slides (optional - remove if not wanted)
// setInterval(nextSlide, 10000); // Advance every 10 seconds