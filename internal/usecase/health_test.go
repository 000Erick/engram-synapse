package usecase_test

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	_ "modernc.org/sqlite" // pure-Go driver; no CGO needed for tests

	"github.com/000Erick/engram-synapse/internal/usecase"
)

func createTempDB(t *testing.T, name string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatalf("createTempDB %s: %v", name, err)
	}
	if _, err := db.Exec("CREATE TABLE _ping (id INTEGER PRIMARY KEY)"); err != nil {
		t.Fatalf("createTempDB ping table: %v", err)
	}
	db.Close()
	return path
}

func TestHealthUsecase_BothDBsReachable(t *testing.T) {
	engramPath := createTempDB(t, "engram.db")
	synapsePath := createTempDB(t, "synapse.db")

	uc := usecase.NewHealthUsecase(engramPath, synapsePath, "test")
	result, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("health.Run: %v", err)
	}
	if !result.EngramReachable {
		t.Error("expected engram_reachable=true")
	}
	if !result.SynapseReachable {
		t.Error("expected synapse_reachable=true")
	}
	if result.Status != "ok" {
		t.Errorf("expected status=ok, got %q", result.Status)
	}
	if result.Version != "test" {
		t.Errorf("expected version=%q, got %q", "test", result.Version)
	}
}

func TestHealthUsecase_EngramUnreachable(t *testing.T) {
	synapsePath := createTempDB(t, "synapse.db")

	uc := usecase.NewHealthUsecase("/nonexistent/engram.db", synapsePath, "test")
	result, err := uc.Run(context.Background())
	if err != nil {
		t.Fatalf("health.Run: %v", err)
	}
	if result.EngramReachable {
		t.Error("expected engram_reachable=false for nonexistent DB")
	}
	if result.Status != "degraded" {
		t.Errorf("expected status=degraded, got %q", result.Status)
	}
}
