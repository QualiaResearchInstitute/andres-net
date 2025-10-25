import type { SimulationState } from "./types";

/**
 * Skillpack scaffolding (Phase-1)
 * - Lightweight registry of "skills" that can be activated to influence the sim or its params.
 * - This is intentionally minimal to establish extension points and deterministic behavior.
 */

export type SkillContext = {
  getSim: () => SimulationState | null;
  // Optional event hook placeholders for future wiring:
  onTick?: (cb: () => void) => () => void;
};

export type Skill = {
  id: string;
  name: string;
  version: string;
  description?: string;
  // Called when enabling the skill; returns a disposer to disable it
  activate: (ctx: SkillContext) => { dispose: () => void };
};

export class SkillpackRegistry {
  private skills = new Map<string, Skill>();
  private active = new Map<string, { dispose: () => void }>();

  register(skill: Skill) {
    if (this.skills.has(skill.id)) {
      throw new Error(`Skill already registered: ${skill.id}`);
    }
    this.skills.set(skill.id, skill);
  }

  list(): Skill[] {
    return [...this.skills.values()];
  }

  get(id: string): Skill | undefined {
    return this.skills.get(id);
  }

  activate(id: string, ctx: SkillContext) {
    if (this.active.has(id)) return; // already active
    const skill = this.skills.get(id);
    if (!skill) throw new Error(`Unknown skill: ${id}`);
    const handle = skill.activate(ctx);
    this.active.set(id, handle);
  }

  deactivate(id: string) {
    const handle = this.active.get(id);
    if (handle) {
      try {
        handle.dispose();
      } finally {
        this.active.delete(id);
      }
    }
  }

  deactivateAll() {
    for (const [id, handle] of this.active) {
      try {
        handle.dispose();
      } catch {}
      this.active.delete(id);
    }
  }
}

/**
 * Example built-in skills (stubs)
 * - These do not alter simulation state yet; they serve as examples for later wiring.
 */

export const CalmNoiseSkill: Skill = {
  id: "calm-noise",
  name: "Calm Noise",
  version: "0.0.1",
  description: "Placeholder for reducing stochasticity/perturbations",
  activate: (ctx: SkillContext) => {
    // Future: periodically adjust sim.noiseSeed or attention gammaD/betaD via callbacks
    const disposer = () => {
      // cleanup subscriptions when present
    };
    return { dispose: disposer };
  },
};

export const FocusPrimaryPlaneSkill: Skill = {
  id: "focus-primary",
  name: "Focus Primary Plane",
  version: "0.0.1",
  description: "Placeholder for biasing attention heads to primary plane",
  activate: (ctx: SkillContext) => {
    // Future: nudge a UI/engine parameter to bind a head to primary plane centroid
    const disposer = () => {};
    return { dispose: disposer };
  },
};

/**
 * Factory to create a default registry with built-in stub skills.
 */
export function makeDefaultSkillpack(): SkillpackRegistry {
  const reg = new SkillpackRegistry();
  reg.register(CalmNoiseSkill);
  reg.register(FocusPrimaryPlaneSkill);
  return reg;
}
