#!/usr/bin/env python3
"""
Tests for DualBasinModel.
Copyright (c) 2025-2026 ForgottenForge.xyz
SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
import pytest
import numpy as np

class TestDualBasinModel:
    def test_import(self):
        from sigma_c.adapters.protein import DualBasinModel
        assert DualBasinModel is not None

    def test_instantiation(self):
        from sigma_c.adapters.protein import DualBasinModel
        model = DualBasinModel(N=15, S=4, contacts=6)
        assert model.N == 15
        assert model.S == 4

    def test_simulate_returns_dict(self):
        from sigma_c.adapters.protein import DualBasinModel
        model = DualBasinModel(N=15, S=4, contacts=6)
        r = model.simulate(alpha=0.3, n_steps=500, n_trials=3)
        assert 'D' in r
        assert 'gamma' in r
        assert 'sigma' in r
        assert 'Q_nat_mean' in r

    def test_sigma_product(self):
        from sigma_c.adapters.protein import DualBasinModel
        model = DualBasinModel(N=15, S=4, contacts=6)
        r = model.simulate(alpha=0.3, n_steps=500, n_trials=3)
        # sigma should equal D * gamma
        assert abs(r['sigma'] - r['D'] * r['gamma']) < 1e-10

    def test_native_contacts_fraction(self):
        from sigma_c.adapters.protein import DualBasinModel
        model = DualBasinModel(N=15, S=4, contacts=6)
        config = model.native_state.copy()  # Perfect native state
        Q = model.native_contacts_fraction(config)
        assert Q == 1.0  # All native contacts formed

    def test_energy_at_native(self):
        from sigma_c.adapters.protein import DualBasinModel
        model = DualBasinModel(N=15, S=4, contacts=6)
        config = model.native_state.copy()
        e = model.energy(config, alpha=0.0)
        assert e <= 0  # Native state at alpha=0 should have negative energy
