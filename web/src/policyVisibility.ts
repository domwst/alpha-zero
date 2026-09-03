export type PolicyVisibility = 'always' | 'network_turn' | 'never';

export function revealsMoveGuidance(
  visibility: PolicyVisibility,
  isNetworkTurn: boolean,
): boolean {
  return visibility === 'always' || (visibility === 'network_turn' && isNetworkTurn);
}
