document.addEventListener("DOMContentLoaded", function () {
  let coll = document.getElementsByClassName("collapsible-container");
  let defaultOpenLines = 10; // Maximum number of lines to display without collapsing

  for (let i = 0; i < coll.length; i++) {
    let trigger = coll[i].querySelector('.collapsible-trigger');
    let content = coll[i].querySelector('.collapsible-content');
    
    if (!trigger || !content) {
      console.error(`Block ${i}: Missing trigger or content element`);
      continue;
    }
    
    // Count lines using the line number element if available
    let codeLines = 0;
    let lineNumberElement = content.querySelector('.lineno');
    
    if (lineNumberElement && lineNumberElement.textContent) {
      // Count actual line numbers
      codeLines = lineNumberElement.textContent.trim().split('\n').length;
    } else if (content.textContent) {
      // Fallback: estimate from content
      codeLines = content.textContent.split('\n').length;
    } else {
      console.warn(`Block ${i}: No text content found`);
      continue;
    }

    console.log(`Block ${i}: ${codeLines} lines`);

    if (codeLines <= defaultOpenLines) {
      trigger.style.display = 'none'; // Hide trigger for short code blocks
      content.style.maxHeight = content.scrollHeight + "px";
    } else {
      trigger.addEventListener("click", function () {
        // Toggle button text
        if (this.innerHTML.includes("Expand")) {
          this.innerHTML = "Collapse";
        } else {
          this.innerHTML = "Expand";
        }
        this.classList.toggle("active");
        if (content.style.maxHeight) {
          content.style.maxHeight = null;
          // Scroll page to trigger element position
          content.scrollIntoView({
            behavior: "smooth",
            block: "center"
          });
        } else {
          content.style.maxHeight = content.scrollHeight + "px";
        }
      });
    }
  }
});
