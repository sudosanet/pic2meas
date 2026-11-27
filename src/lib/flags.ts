// URL switches for comparing versions in evaluations: ?noart (no drawing-trained keypoints), ?nofit (no proportion fit).
const q = typeof location === "undefined" ? new URLSearchParams() : new URLSearchParams(location.search);
export const FLAGS = { art: !q.has("noart"), fit: !q.has("nofit") };
