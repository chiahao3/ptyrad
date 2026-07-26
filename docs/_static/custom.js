// Add this to _static/custom.js
document.addEventListener('DOMContentLoaded', function() {
    // Wait a bit for tabs to be fully initialized
    setTimeout(function() {
        // Map reference targets to tab configurations
        const tabMappings = {
            'install-nvidia': 'NVidia',
            'install-amd': 'AMD',
            'install-cpu': 'CPU',
            'install-macos': 'MacOS'
        };
        
        // Function to activate a specific tab by simulating a click
        function activateTabByClick(tabName) {
            // Find all tab labels
            const tabLabels = document.querySelectorAll('.sd-tab-label');
            
            for (let label of tabLabels) {
                // Check if this label matches our target tab name
                if (label.textContent.trim() === tabName) {
                    // Simulate a click on the tab label
                    label.click();
                    return true;
                }
            }
            return false;
        }
        
        // Add click handlers to reference links
        const refLinks = document.querySelectorAll('a[href^="#install-"]');
        refLinks.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Get the target reference
                const targetRef = this.getAttribute('href').substring(1);
                const targetTab = tabMappings[targetRef];
                
                if (targetTab) {
                    // Activate the target tab
                    if (activateTabByClick(targetTab)) {
                        // Scroll to the installation options section after a short delay
                        setTimeout(() => {
                            const installSection = document.querySelector('.sd-tab-set');
                            if (installSection) {
                                installSection.scrollIntoView({ 
                                    behavior: 'smooth', 
                                    block: 'start' 
                                });
                            }
                        }, 200);
                    }
                }
            });
        });
        
    }, 500); // Wait 500ms for tabs to initialize
});