function goBack() {
  window.history.back();
}

document.addEventListener("DOMContentLoaded", function () {
  const playBtn = document.getElementById("playButton");
  const audio = document.getElementById("motivasiAudio");
  const icon = playBtn.querySelector("i");

  let isPlaying = false;

  playBtn.addEventListener("click", function () {
    if (!isPlaying) {
      audio.play().then(() => {
        icon.setAttribute("data-lucide", "pause");
        lucide.createIcons(); // Membuat ulang ikon setelah diubah
        isPlaying = true;
      }).catch((error) => {
        console.error("Error playing audio:", error);
      });
    } else {
      audio.pause();
      icon.setAttribute("data-lucide", "play");
      lucide.createIcons(); // Membuat ulang ikon setelah diubah
      isPlaying = false;
    }
  });

  audio.addEventListener("ended", () => {
    icon.setAttribute("data-lucide", "play");
    lucide.createIcons();
    isPlaying = false;
  });
});




