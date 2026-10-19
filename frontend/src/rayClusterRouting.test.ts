import { describe, expect, it } from "vitest";
import { routeClusterLinks } from "./rayClusterRouting";

const bounds = {
  linkBandTop: 8,
  linkBandBottom: 32,
  controlRowTop: 36,
  controlRowBottom: 120,
  gutterTop: 120,
  gutterBottom: 132,
  workerRowTop: 132,
  workerRowBottom: 220,
};

describe("routeClusterLinks", () => {
  it("routes same-row control links through the link band via top centers", () => {
    const paths = routeClusterLinks(
      [
        {
          key: "a-b",
          healthy: true,
          sourceRow: "control",
          targetRow: "control",
          sx: 100,
          scx: 80,
          sourceTop: 40,
          sourceBottom: 110,
          ex: 198,
          tx: 210,
          targetTop: 40,
          targetBottom: 110,
        },
      ],
      bounds,
    );
    expect(paths[0].d).toContain("M 80.0 40.0");
    expect(paths[0].d).toMatch(/L 210\.0 40\.0$/);
    expect(paths[0].d).not.toMatch(/L 198\.0/);
    expect(paths[0].d).not.toMatch(/L 100\.0/);
  });

  it("routes control to worker through the gutter", () => {
    const paths = routeClusterLinks(
      [
        {
          key: "c-w",
          healthy: true,
          sourceRow: "control",
          targetRow: "worker",
          sx: 200,
          scx: 180,
          sourceTop: 40,
          sourceBottom: 115,
          ex: 300,
          tx: 320,
          targetTop: 135,
          targetBottom: 210,
        },
      ],
      bounds,
    );
    expect(paths[0].d).toMatch(/115\.0/);
    expect(paths[0].d).toMatch(/135\.0/);
    expect(paths[0].d).toMatch(/126\.0/);
  });

  it("keeps worker same-row links above worker cards", () => {
    const paths = routeClusterLinks(
      [
        {
          key: "w-w",
          healthy: true,
          sourceRow: "worker",
          targetRow: "worker",
          sx: 100,
          scx: 80,
          sourceTop: 140,
          sourceBottom: 210,
          ex: 198,
          tx: 210,
          targetTop: 140,
          targetBottom: 210,
        },
      ],
      bounds,
    );
    expect(paths[0].d).not.toMatch(/20\.0/);
    expect(paths[0].d).toContain("M 80.0 140.0");
    expect(paths[0].d).toMatch(/L 210\.0 140\.0$/);
  });
});
