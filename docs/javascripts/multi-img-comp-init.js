document.addEventListener('DOMContentLoaded', function() {
  for (const e of document.querySelectorAll(".b-dics")) {
    new Dics({ container: e, textPosition: "bottom" });
  }
});
